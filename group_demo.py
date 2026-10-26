import os
import torch
import time
import torch.distributed as dist
# This creates default group (WORLD)
def main():
  # dist.init_process_group(backend='gloo')
  dist.init_process_group(backend="gloo", init_method="env://") 
  print(f"rank={os.environ['RANK']} world={os.environ['WORLD_SIZE']} "
      f"group_rank={os.environ['GROUP_RANK']} "
      f"restart={os.environ.get('TORCHELASTIC_RESTART_COUNT')}", flush=True)
  rank = dist.get_rank()
  print(f"rank={rank}")


  time.sleep(30)
  # Every rank must execute these lines in the same order
  # These are basically used for 
  group_a = dist.new_group([0,1])
  group_b = dist.new_group([2,3])

  tensor = torch.tensor([float(rank)])
  if rank in (0,1):
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group_a)
  else:
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group_b)

  print(f"rank {rank}: t = {tensor.item()}")

  dist.destroy_process_group()

if  __name__ == "__main__":
  main()
