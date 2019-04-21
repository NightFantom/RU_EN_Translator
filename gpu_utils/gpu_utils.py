import os
import torch
import numpy as np
import translator_constants.global_constant as glc


def get_freer_gpu():
    temp_file_name = os.path.join(glc.BASE_PATH, "gpu_memory")
    os.system(f'nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >{temp_file_name}')
    with open(temp_file_name, 'r') as file:
        info = file.readlines()
    memory_available = [int(x.split()[2]) for x in info]
    return np.argmax(memory_available)


def get_device():
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            index = get_freer_gpu()
            device_name = f"cuda:{index}"
        else:
            device_name = "cuda"
    else:
        device_name = "cpu"
    return torch.device(device_name)
