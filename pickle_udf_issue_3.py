import os
import torch.distributed.rpc as rpc

def say_hello(msg):
    print(f"[rank {os.environ['RANK']}] RUNNING:", msg, flush=True)
    return msg

class Demo:
    def __reduce__(self):
        return (say_hello, ("sent by rank 0",))

def echo(x):
    return x

def main():
    rank = int(os.environ["RANK"])
    rpc.init_rpc(f"worker{rank}", rank=rank, world_size=2)
    if rank == 0:
        result = rpc.rpc_sync("worker1", echo, args=(Demo(),))
        print("rank 0 got back:", result)
    rpc.shutdown()

main()
