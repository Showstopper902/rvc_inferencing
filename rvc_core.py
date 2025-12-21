# rvc_core.py
import os, torch, warnings, traceback
from fairseq import checkpoint_utils
from vc_infer_pipeline import VC
from infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from my_utils import load_audio
from scipy.io import wavfile          # (kept for compatibility, even if unused)
from config import Config
import soundfile as sf

# -----------------------------
# Global config / state
# -----------------------------
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


# -----------------------------
# HuBERT loader
# -----------------------------
def load_hubert():
    global hubert_model
    if hubert_model is not None:
        return
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(["hubert_base.pt"])
    hubert_model = models[0].to(config.device)
    hubert_model = hubert_model.half() if config.is_half else hubert_model.float()
    hubert_model.eval()


# -----------------------------
# Decoder compatibility check
# -----------------------------
def _assert_decoder_compatible(net_g, weight: dict):
    """
    Make sure NSF decoder shapes match between checkpoint and model.
    If they don't, we *hard fail* instead of silently skipping,
    because that leads to static / pitch noise.
    """
    model_sd = net_g.state_dict()
    bad = []

    critical_prefixes = (
        "dec.noise_convs.",
        "dec.ups.",
    )
    critical_exact = ("dec.conv_post.weight",)

    for k, v in weight.items():
        is_critical = k.startswith(critical_prefixes) or k in critical_exact
        if not is_critical:
            continue
        if k in model_sd and model_sd[k].shape != v.shape:
            bad.append((k, v.shape, model_sd[k].shape))

    if bad:
        msg = "[rvc_core] FATAL: decoder NSF shape mismatch detected:\n"
        for k, s_ckpt, s_model in bad:
            msg += f"  - {k}: ckpt {s_ckpt} vs model {s_model}\n"
        msg += (
            "This checkpoint was not trained with the same decoder architecture "
            "as this inferencer. You must retrain or use matching pretraineds."
        )
        raise RuntimeError(msg)


# -----------------------------
# Safe weight loader (logs skips)
# -----------------------------
def _safe_load_weights(net_g, weight: dict):
    model_sd = net_g.state_dict()
    loaded = 0
    skipped = 0
    mismatched = []

    for k, v in weight.items():
        if k in model_sd:
            if model_sd[k].shape == v.shape:
                model_sd[k] = v
                loaded += 1
            else:
                mismatched.append((k, v.shape, model_sd[k].shape))
                skipped += 1
        else:
            skipped += 1

    net_g.load_state_dict(model_sd, strict=False)

    total = len(model_sd)
    print(
        f"[rvc_core] safe_load: loaded {loaded} params, "
        f"skipped {skipped} mismatched/extra params, total_model_params={total}"
    )
    if mismatched:
        print("[rvc_core] Skipped keys due to shape mismatch (showing up to 10):")
        for k, s_ckpt, s_model in mismatched[:10]:
            print(f"  - {k}: ckpt {s_ckpt} vs model {s_model}")


# -----------------------------
# Architecture detection
# -----------------------------
def _detect_arch_from_weight(weight: dict) -> str:
    """
    True rule (your setup, with these pretraineds):

      v1-style  → enc_p.emb_phone.weight: [192, 768]
      v2-style  → enc_p.emb_phone.weight: [192, 256]
    """
    w = weight.get("enc_p.emb_phone.weight")
    if isinstance(w, torch.Tensor):
        shape = tuple(w.shape)
        print(f"[rvc_core] enc_p.emb_phone.weight shape={shape}")
        if shape == (192, 768):
            return "768"  # “v1-style”, 768 upstream projection
        elif shape == (192, 256):
            return "256"  # “v2-style”, 256 upstream projection
    # Fallback
    return "768"


# -----------------------------
# Core: load VC model
# -----------------------------
def get_vc(weight_path, sid=0):
    global tgt_sr, net_g, vc, cpt, version

    print(f"Loading model: {weight_path}")
    cpt = torch.load(weight_path, map_location="cpu")

    if "config" not in cpt:
        raise KeyError(f"{weight_path} has no 'config' key; export_rvc_model must be used.")

    cfg = cpt["config"]
    if isinstance(cfg, (list, tuple)):
        cfg = list(cfg)
    else:
        raise TypeError("Expected list-like config in checkpoint.")

    # sample rate is last element
    tgt_sr = cfg[-1]

    # update spk_embed_dim from emb_g.weight
    weight = cpt["weight"]
    spk_dim = weight["emb_g.weight"].shape[0]
    cfg[-3] = spk_dim  # spk_embed_dim in 18-param config
    print(f"[rvc_core] spk_embed_dim={spk_dim}")

    # Detect architecture from weights
    arch = _detect_arch_from_weight(weight)
    print(f"[rvc_core] choosing arch={arch}")

    use_f0 = int(cpt.get("f0", 1))
    version = str(cpt.get("version", arch))

    # Instantiate correct class
    if arch == "256":
        # “v2-style” (192,256)
        if use_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cfg, is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cfg)
    else:
        # default to 768 “v1-style”
        if use_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cfg, is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cfg)

    # Make sure decoder is truly compatible BEFORE loading weights
    _assert_decoder_compatible(net_g, weight)

    # Remove enc_q for inference (original behavior)
    if hasattr(net_g, "enc_q"):
        del net_g.enc_q

    # Load weights with logging
    _safe_load_weights(net_g, weight)

    net_g.eval().to(config.device)
    net_g = net_g.half() if config.is_half else net_g.float()

    vc = VC(tgt_sr, config)
    return vc


# -----------------------------
# Single-file inference helper
# -----------------------------
def vc_single(
    sid,
    input_audio,
    f0_up_key,
    f0_file,
    f0_method,
    file_index,
    index_rate,
    crepe_hop_length,
    output_path=None,
):
    global tgt_sr, net_g, vc, hubert_model

    if input_audio is None:
        raise ValueError("No input audio file provided.")

    try:
        audio = load_audio(input_audio, 16000)

        if hubert_model is None:
            load_hubert()

        # normalize FAISS index naming
        file_index = file_index.strip().replace("trained", "added")
        times = [0, 0, 0]

        print(f"Using the following f0 method: {f0_method}")
        audio_opt = vc.pipeline(
            hubert_model,
            net_g,
            sid,
            audio,
            times,
            int(f0_up_key),
            f0_method,
            file_index,
            index_rate,
            cpt.get("f0", 1),
            version,
            crepe_hop_length,
            None,
        )

        if output_path:
            sf.write(output_path, audio_opt, tgt_sr, format="WAV")
        print(f"Inference complete: {output_path}")
    except Exception:
        print("❌ Inference failed:")
        traceback.print_exc()
