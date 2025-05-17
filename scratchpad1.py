# scratchpad1.py

import torch
from termcolor import colored


DIM_0_SIZE = 60614


coords = torch.randint(low=0, high=5000, size=(DIM_0_SIZE, 4), dtype=torch.int32)

batch_mask = torch.randint(0, 2, (DIM_0_SIZE,), dtype=torch.bool)


this_coords = coords[batch_mask, :]

print('\n' + 'this_coords: ')
print(type(this_coords))
print(this_coords.dtype)
print(this_coords.shape)

print('\n')


