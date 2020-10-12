import numpy as np
_gpu_available = False
try:
    import cupy as cp
    _gpu_available = True
except ImportError:
    pass


class GPUWrapper():
    def __init__(self):
        self.xnp = np
        self.use_gpu = False
        if _gpu_available:
            self.xnp = cp
            self.use_gpu = True

    def use_gpu_acceleration(self, use_gpu):
        if use_gpu:
            if(_gpu_available):
                self.xnp = cp
                self.use_gpu = True
            else:
                print("Attempt to use GPU acceleration failed. Verify that "
                      "CuPy is properly installed and configured")
        else:
            self.xnp = np
            self.use_gpu = False


gpu_wrapper = GPUWrapper()


def use_gpu_acceleration(use_gpu):
    gpu_wrapper.use_gpu_acceleration(use_gpu)
