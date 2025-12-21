import torch
from multiprocessing import cpu_count


class Config:
    def __init__(self):
        self.device = "cuda:0"
        self.is_half = True
        self.n_cpu = 0
        self.gpu_name = None
        self.gpu_mem = None

        # Static defaults (no argparse)
        self.python_cmd = "python"
        self.listen_port = 7865
        self.iscolab = False
        self.noparallel = False
        self.noautoopen = True
        self.use_gfloat = False
        self.paperspace = False

        if self.use_gfloat:
            print("Using g_float instead of g_half")
            self.is_half = False

        # Continue normal GPU/device configuration
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    def device_config(self) -> tuple:
        if torch.cuda.is_available():
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
            if (
                ("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
                or "P40" in self.gpu_name.upper()
                or "1060" in self.gpu_name
                or "1070" in self.gpu_name
                or "1080" in self.gpu_name
            ):
                print("16系/10系显卡和P40强制单精度")
                self.is_half = False
                with open("trainset_preprocess_pipeline_print.py", "r") as f:
                    strr = f.read().replace("3.7", "3.0")
                with open("trainset_preprocess_pipeline_print.py", "w") as f:
                    f.write(strr)
            else:
                self.gpu_name = None

            self.gpu_mem = int(
                torch.cuda.get_device_properties(i_device).total_memory
                / 1024
                / 1024
                / 1024
                + 0.4
            )

            if self.gpu_mem <= 4:
                with open("trainset_preprocess_pipeline_print.py", "r") as f:
                    strr = f.read().replace("3.7", "3.0")
                with open("trainset_preprocess_pipeline_print.py", "w") as f:
                    f.write(strr)

        elif torch.backends.mps.is_available():
            print("No supported Nvidia cards found, using MPS for inference")
            self.device = "mps"
        else:
            print("No supported Nvidia cards found, using CPU for inference")
            self.device = "cpu"
            # Fork Feature: Force g_float (is_half = False) if --use_gfloat arg is used.
            if not self.use_gfloat:
                self.is_half = False

        if self.n_cpu == 0:
            self.n_cpu = cpu_count()

        # Adjust padding and query sizes depending on precision and VRAM
        if self.is_half:
            x_pad, x_query, x_center, x_max = 3, 10, 60, 65  # 6 GB config
        else:
            x_pad, x_query, x_center, x_max = 1, 6, 38, 41  # 5 GB config

        if self.gpu_mem is not None and self.gpu_mem <= 4:
            x_pad, x_query, x_center, x_max = 1, 5, 30, 32

        return x_pad, x_query, x_center, x_max
