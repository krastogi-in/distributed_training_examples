"""Connect to an existing TCPStore as a non-master client.

Any process that can reach host:port can call the same APIs — there is no
auth on the store. Point this only at your own test jobs (e.g. group_demo
started with torchrun on localhost).
"""
import argparse

import torch.distributed as dist


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=29501)
    args = p.parse_args()

    store = dist.TCPStore(args.host, args.port, is_master=False)
    print(f"connected to {args.host}:{args.port}")
    print(f"num_keys() = {store.num_keys()}")

    print(store.list_keys())

    x = store.list_keys()
    print(store.get(x[0]))
    print(store.delete_key(x[0]))
    store.set("foreign_client", b"hello-from-outside")
    print("set('foreign_client', b'hello-from-outside')")
    print(f"num_keys() = {store.num_keys()}")


if __name__ == "__main__":
    main()
