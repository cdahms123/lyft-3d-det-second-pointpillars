# scratchpad1.py

import torch
from termcolor import colored


# # check GPU availability
# if torch.cuda.is_available():
#     device = 'cuda'
#     print(colored('\n' + 'using GPU' + '\n', 'green'))
# else:
#     device = 'cpu'
#     print(colored('\n' + 'GPU does not seem to be available, using CPU' + '\n', 'red'))
# # end if


# print('\n' + 'device: ')
# print(type(device))
# print(device)

# print('\n')




# check GPU availability
if torch.cuda.is_available():
    device = torch.device("cuda")
    # device = torch.device("cuda:0")
    print(colored('\n' + 'using GPU' + '\n', 'green'))
else:
    device = torch.device("cpu:0")
    print(colored('\n' + 'GPU does not seem to be available, using CPU' + '\n', 'red'))
# end if




print('\n' + 'device: ')
print(type(device))
print(device)

print('\n')








