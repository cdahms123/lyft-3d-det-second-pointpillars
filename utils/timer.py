# timer.py

import time
import contextlib
from contextlib import contextmanager
import torch

@contextmanager
def simple_timer(name=''):
    t = time.time()
    yield 
    print(f"{name} exec time: {time.time() - t}")
# end function

@contextlib.contextmanager
def torch_timer(name=''):
    torch.cuda.synchronize()
    t = time.time()
    yield
    torch.cuda.synchronize()
    print(name, "time:", time.time() - t)
# end function


