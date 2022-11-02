
import torch
from torch.utils.data import RandomSampler, DistributedSampler, SequentialSampler

from utils.configurable import configurable
from utils.register import Registry
from utils.dist import get_world_size, get_rank

DATASET_REGISTRY = Registry("Datasets")

def build_dataset(args):
    dataset = DATASET_REGISTRY.get(args.dataset)(args)
    train_data, val_data = dataset.get_data()
    n_classes = dataset.n_classes
    return train_data, val_data, n_classes


def _cfg_to_trainloader(args):
    return {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "drop_last": args.drop_last,

        "distributed": args.distributed,
        "world_size": get_world_size(),
        "rank": get_rank(),
    }


@configurable(from_config=_cfg_to_trainloader)
def build_train_dataloader(
    train_dataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool,
    distributed: bool,
    world_size: int,
    rank: int,
):
    if distributed:
        sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
    else:
        sampler = RandomSampler(train_dataset)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    return train_loader



def _cfg_to_valloader(args):
    return {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,

        "distributed": args.distributed,
        "distributed_val": args.distributed_val,
        "world_size": get_world_size(),
        "rank": get_rank(),
    }

@configurable(from_config=_cfg_to_valloader)
def build_val_dataloader(
    val_dataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    distributed: bool,
    distributed_val: bool,
    world_size: int,
    rank: int,
):
    if distributed and distributed_val:
        if len(val_dataset) % world_size != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
        sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        sampler = SequentialSampler(val_dataset)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return val_loader