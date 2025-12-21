# rvc_core.py
import os, torch, warnings
from fairseq import checkpoint_utils
from vc_infer_pipeline import VC
from infer_pack.models import SynthesizerTrnMs256NSFsid, SynthesizerTrnMs256NSFsid_nono
from infer_pack.modelsv2 import SynthesizerTrnMs768NSFsid, SynthesizerTrnMs768NSFsid_nono
from my_utils import load_audio
from scipy.io import wavfile
from config import Config
import soundfile as sf
import traceback

config = Config()
device = config.device
is_half = config.is_half
hubert_model = None
cpt = None
net_g = None
tgt_sr = None
vc = None
version = None

warnings.filterwarnings("ignore")
torch.manual_seed(114514)

def load_hubert():
    global hubert_model
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(["hubert_base.pt"])
    hubert_model = models[0].to(config.device)
    hubert_model = hubert_model.half() if config.is_half else hubert_model.float()
    hubert_model.eval()

def get_vc(weight_path, sid=0):
    global n_spk, tgt_sr, net_g, vc, cpt, version

    print(f"Loading model: {weight_path}")
    cpt = torch.load(weight_path, map_location="cpu")

    cfg = cpt["config"]
    tgt_sr = cfg[-1]

    # update spk_embed_dim
    cfg[-3] = cpt["weight"]["emb_g.weight"].shape[0]

    # --------------------------------------------------------
    # 🔍 AUTO-DETECT ARCHITECTURE USING inter_channels
    # cfg[2] will be:
    #   192 → small-arch (256 model class)
    #   256 → big-arch (768 model class)
    # --------------------------------------------------------
    inter_channels = cfg[2]
    if inter_channels == 192:
        detected_arch = "small"
    else:
        detected_arch = "big"

    print(f"[rvc_core] Detected architecture: {detected_arch} (inter_channels={inter_channels})")

    version = cpt.get("version", "v1")
    use_f0 = cpt.get("f0", 1)

    # --------------------------------------------------------
    # 🔥 ARCH-AWARE INSTANTIATION
    # small → 256NSFsid classes
    # big   → 768NSFsid classes
    # --------------------------------------------------------
    if detected_arch == "small":
        if use_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cfg, is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cfg)
    else:
        if use_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cfg, is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cfg)

    # original logic
    del net_g.enc_q
    net_g.load_state_dict(cpt["weight"], strict=False)
    net_g.eval().to(config.device)
    net_g = net_g.half() if config.is_half else net_g.float()

    vc = VC(tgt_sr, config)
    return vc


def vc_single(sid, input_audio, f0_up_key, f0_file, f0_method, file_index, index_rate, crepe_hop_length, output_path=None):
    global tgt_sr, net_g, vc, hubert_model
    if input_audio is None:
        raise ValueError("No input audio file provided.")
    try:
        audio = load_audio(input_audio, 16000)
        if hubert_model is None:
            load_hubert()
        file_index = file_index.strip().replace("trained", "added")
        times = [0, 0, 0]
        audio_opt = vc.pipeline(
            hubert_model, net_g, sid, audio, times,
            int(f0_up_key), f0_method, file_index, index_rate,
            cpt.get("f0", 1), version, crepe_hop_length, None
        )
        if output_path:
            sf.write(output_path, audio_opt, tgt_sr, format="WAV")
        print(f"Inference complete: {output_path}")
    except Exception as e:
        print("❌ Inference failed:")
        traceback.print_exc()
