def distprint(x, local_rank=None):
    if local_rank == 0:
        print(x, flush=True)