import torch
import torch.distributed as dist
import torch.nn.functional as F
import os
import wandb

def ddp_setup():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", dist.get_rank())))

def gather_losses_and_report(epoch_losses_dict, other_metrics, rank, world_size, split='train', num_steps=None, dry_run=True):
    if world_size > 1:
        epoch_losses_dict = {k: torch.stack(v) for k, v in epoch_losses_dict.items()}
        
        if rank == 0:
            gathered_losses_dicts = {loss_name: [torch.empty_like(loss_tensor) for _ in range(world_size)] for loss_name, loss_tensor in epoch_losses_dict.items()}
        else:
            gathered_losses_dicts = {loss_name: None for loss_name in epoch_losses_dict.keys()}

        for loss_name, loss_tensor in epoch_losses_dict.items():
            dist.gather(loss_tensor, gathered_losses_dicts[loss_name])
    else:
        gathered_losses_dicts = epoch_losses_dict

    to_report = None
    if rank == 0:
        to_report = compute_metric_means(gathered_losses_dicts, split=split, num_steps=num_steps)
        to_report.update(other_metrics)
        if not dry_run:
            wandb.log(to_report)
    
    del gathered_losses_dicts

    return to_report

def compute_metric_means(loss_dict, split='train', num_steps=None):
    # losses shape: [world_size, num_steps]
    return {f"{split}/{loss_name}": torch.stack(loss_arr).cpu().mean().item() for loss_name, loss_arr in loss_dict.items()}

def accuracy(pred, labels):
    pred = pred.squeeze()
    labels = labels.squeeze()
    if len(pred.shape) == 1:
        pred = torch.round(F.sigmoid(pred))
    else:
        pred = torch.argmax(pred, dim=1)
    return (pred == labels).sum() / len(pred)