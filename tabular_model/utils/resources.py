import torch


def gpu_device_name():
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        return f"cuda:{device_id}"
    else:
        return "cpu"


def device_name(gpu=False):
    return gpu_device_name() if gpu else 'cpu'
