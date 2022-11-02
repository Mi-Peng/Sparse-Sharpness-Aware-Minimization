import torch.optim as optim

from utils.register import Registry

OPTIMIZER_REGISTRY = Registry("Optimizer")
LR_SCHEDULER_REGISTRY = Registry("LRscheduler")


def build_base_optimizer(args, parameters):
    opt_kwargs = dict(
        lr=args.lr,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov,
    )
    if getattr(args, 'momentum', None) is not None:
        opt_kwargs['momentum'] = args.momentum
    if getattr(args, 'eps', None) is not None:
        opt_kwargs['eps'] = args.eps
    if getattr(args, 'betas', None) is not None:
        opt_kwargs['betas'] = args.betas

    # Parse Optimizer Args
    args_opt = args.opt.split('-') # `ssamf-sgd` or `sgd`
    if len(args_opt) == 1:
        base_opt = args_opt[0]
    elif len(args_opt) == 2:
        sam_opt, base_opt = args_opt[0], args_opt[1]

    # Build Base Optimizer
    if base_opt == 'sgd':
        base_optimizer = optim.SGD(params=parameters, **opt_kwargs)
    elif base_opt == 'adamw':
        base_optimizer = optim.AdamW(params=parameters, **opt_kwargs)
    else:
        raise ValueError("Incorrect base optimizer.")
    
    return base_optimizer

def build_optimizer(args, model):
    base_optimizer = build_base_optimizer(args, model.parameters())
    args_opt = args.opt.split('-') # ssamf-sgd
    if len(args_opt) == 1:
        optimizer = base_optimizer
    elif len(args_opt) == 2:
        sam_opt, base_opt = args_opt[0], args_opt[1]
        optimizer = OPTIMIZER_REGISTRY.get(sam_opt.upper())(params=model.parameters(), base_optimizer=base_optimizer, args=args)
    return optimizer, base_optimizer

def build_lr_scheduler(args, optimizer):
    lr_scheduler = LR_SCHEDULER_REGISTRY.get(args.lr_scheduler)(optimizer=optimizer, args=args)
    return lr_scheduler