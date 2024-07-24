import time
import torch
import pynvml

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)


def get_gpu_usage():
    """Returns the current GPU usage in percentage."""
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    mem_used = mem_info.used / mem_info.total * 100
    return mem_used


def increase_gpu_usage(target_usage):
    """Performs GPU computations to increase usage up to the target."""
    while get_gpu_usage() < target_usage:
        # Perform a simple tensor operation to increase GPU usage
        a = torch.rand((10000, 10000), device='cuda')
        b = torch.mm(a, a)
        time.sleep(1)  # Sleep to allow monitoring


def keep_gpu_usage(target_usage):
    if value := get_gpu_usage() < target_usage:
        print('current gpu usage is : ', value)
        print('increase to : ', target_usage)

        increase_gpu_usage(target_usage)