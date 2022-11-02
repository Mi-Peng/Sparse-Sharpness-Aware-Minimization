import os
import time
import datetime

import torch

from models.build import build_model
from data.build import build_dataset, build_train_dataloader, build_val_dataloader
from solver.build import build_optimizer, build_lr_scheduler

from utils.logger import Logger
from utils.dist import init_distributed_model, is_main_process
from utils.seed import setup_seed
from utils.engine import train_one_epoch, evaluate

def main(args):
    # init seed
    setup_seed(args)

    # init dist
    init_distributed_model(args)

    # init log
    logger = Logger(args)
    logger.log(args)

    # build dataset and dataloader
    train_data, val_data, n_classes = build_dataset(args)
    train_loader = build_train_dataloader(
        train_dataset=train_data,
        args=args
    )
    val_loader = build_val_dataloader(
        val_dataset=val_data,
        args=args
    )
    args.n_classes = n_classes
    logger.log(f'Train Data: {len(train_data)}, Test Data: {len(val_data)}.')

    # build model
    model = build_model(args)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    logger.log(f'Model: {args.model}')

    # build loss
    criterion = torch.nn.CrossEntropyLoss()

    # build solver
    optimizer, base_optimizer = build_optimizer(args, model=model_without_ddp)
    lr_scheduler = build_lr_scheduler(args, optimizer=base_optimizer)
    logger.log(f'Optimizer: {type(optimizer)}')
    logger.log(f'LR Scheduler: {type(lr_scheduler)}')

    # resume
    if args.resume:
        checkpoint = torch.load(args.resume_path, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        lr_scheduler.step(args.start_epoch)
        logger.log(f'Resume training from {args.resmue_path}.')


    # start train:
    logger.log(f'Start training for {args.epochs} Epochs.')
    start_training = time.time()
    max_acc = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        start_epoch = time.time()
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model=model, 
            train_loader=train_loader, 
            criterion=criterion, 
            optimizer=optimizer, 
            epoch=epoch, 
            logger=logger, log_freq=args.log_freq, use_closure=(args.opt[:4] == 'sam-' or args.opt[:4] == 'ssam'), 
        )
        lr_scheduler.step(epoch)
        val_stats = evaluate(model, val_loader)

        if max_acc < val_stats["test_acc1"]:
            max_acc = val_stats["test_acc1"]
            if is_main_process:
                torch.save({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, os.path.join(args.output_dir, args.output_name, 'checkpoint.pth'))
        
        logger.wandb_log(epoch=epoch, **train_stats)
        logger.wandb_log(epoch=epoch, **val_stats)
        msg = ' '.join([
            'Epoch:{epoch}',
            'Train Loss:{train_loss:.4f}',
            'Train Acc1:{train_acc1:.4f}',
            'Train Acc5:{train_acc5:.4f}',
            'Test Loss:{test_loss:.4f}',
            'Test Acc1:{test_acc1:.4f}(Max:{max_acc:.4f})',
            'Test Acc5:{test_acc5:.4f}',
            'Time:{epoch_time:.3f}s'])
        logger.log(msg.format(epoch=epoch, **train_stats, **val_stats, max_acc=max_acc, epoch_time=time.time()-start_epoch))
    logger.log('Train Finish. Max Test Acc1:{:.4f}'.format(max_acc))
    end_training = time.time()
    used_training = str(datetime.timedelta(seconds=end_training-start_training))
    logger.log('Training Time:{}'.format(used_training))
    logger.mv('{}_{:.4f}'.format(logger.logger_path, max_acc))


if __name__ == '__main__':
    from configs.defaulf_cfg import default_parser
    cfg_file = default_parser()
    args = cfg_file.get_args()
    main(args)