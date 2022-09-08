import torch


def gpu_device_name():
    """
    :return: CUDA device name if available. Returns 'cpu' otherwise
    """
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        return f"cuda:{device_id}"
    else:
        return "cpu"


def device_name(gpu=False):
    """
    :param gpu: Boolean for whether to use cpu or gpu tensors
    :return: name of the tensors device based on the selected <gpu> option
    """
    return gpu_device_name() if gpu else 'cpu'
