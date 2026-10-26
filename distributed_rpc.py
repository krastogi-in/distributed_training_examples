import os
import time

import torch.distributed.rpc as rpc


def add_one(x):
    return x + 1


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rpc.init_rpc(f"worker{rank}", rank=rank, world_size=world_size)
    print(f"Worker {rank} initialized")
    time.sleep(30)
    if rank > 0:
        result = rpc.rpc_sync(f"worker{rank - 1}", add_one, args=(1,))
        print(f"Worker {rank} result: {result}")
    else:
        print(f"Worker {rank} waiting for RPCs")
    rpc.shutdown()

if __name__ == "__main__":
    main()