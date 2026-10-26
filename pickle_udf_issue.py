import pickletools
from torch.distributed.rpc.internal import _internal_rpc_pickler
import torch

data, tensors = _internal_rpc_pickler.serialize((torch.add, (torch.ones(2), 1), {}))
pickletools.dis(data)   # shows GLOBAL/REDUCE opcodes and which names are referenced
print(len(tensors))     # tensors are pulled out of the pickle bytes
