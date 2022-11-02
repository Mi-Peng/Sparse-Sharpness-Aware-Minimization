import argparse

class default_parser:
    def __init__(self) -> None:
        pass
    
    def wandb_parser(self):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--wandb', action='store_true')
        parser.add_argument('--wandb_project', type=str, default='NeurIPs2022-Sparse SAM', help="Project name in wandb.")
        parser.add_argument('--wandb_name', type=str, default='Default', help="Experiment name in wandb.")
        return parser

    def base_parser(self):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--output_dir', type=str, default='logs', help='Name of dir where save all experiments.')
        parser.add_argument('--output_name', type=str, default=None, help="Name of dir where save the log.txt&ckpt.pth of this experiment. (None means auto-set)")
        parser.add_argument('--resume', action='store_true', help="resume model,opt,etc.")
        parser.add_argument('--resume_path', type=str, default='.')

        parser.add_argument('--seed', type=int, default=1234)
        parser.add_argument('--log_freq', type=int, default=10, help="Frequency of recording information.")

        parser.add_argument('--start_epoch', type=int, default=0)
        parser.add_argument('--epochs', type=int, default=200, help="Epochs of training.")
        return parser

    def dist_parser(self):
        parser = argparse.ArgumentParser(add_help=False)    
        parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
        return parser


    def data_parser(self):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--dataset', type=str, default='CIFAR10_base', help="Dataset name in `DATASETS` registry.")
        parser.add_argument('--datadir', type=str, default='/public/data0/DATA-1/users/mipeng7/datasets', help="Path to your dataset.")
        parser.add_argument('--batch_size', type=int, default=128, help="Batch size used in training and validation.")
        parser.add_argument('--num_workers', type=int, default=8, help="Number of CPU threads for dataloaders.")
        parser.add_argument('--pin_memory', action='store_true', default=True)
        parser.add_argument('--drop_last', action='store_true', default=True)
        parser.add_argument('--distributed_val', action='store_true', help="Enabling distributed evaluation (Only works when use multi gpus).")
        return parser

    def base_opt_parser(self):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--opt', type=str, default='sgd')
        parser.add_argument('--lr', type=float, default=0.05)
        parser.add_argument('--weight_decay', type=float, default=5e-4)
        # sgd
        parser.add_argument('--momentum', type=float, default=0.9, help="Momentum for SGD.(None means the default in optm)")
        parser.add_argument('--nesterov', action="store_true")
        # adam
        parser.add_argument('--betas', type=float, default=None, nargs='+', help="Betas for AdamW Optimizer.(None means the default in optm)")
        parser.add_argument('--eps', type=float, default=None, help="Epsilon for AdamW Optimizer.(None means the default in optm)")
        return parser

    def sam_opt_parser(self):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--rho', type=float, default=0.05, help="Perturbation intensity of SAM type optims.")
        parser.add_argument('--sparsity', type=float, default=0.2, help="The proportion of parameters that do not calculate perturbation.")
        parser.add_argument('--update_freq', type=int, default=5, help="Update frequency (epoch) of sparse SAM.")

        parser.add_argument('--num_samples', type=int, default=1024, help="Number of samples to compute fisher information. Only for `ssam-f`.")
        parser.add_argument('--drop_rate', type=float, default=0.5, help="Death Rate in `ssam-d`. Only for `ssam-d`.")
        parser.add_argument('--drop_strategy', type=str, default='gradient', help="Strategy of Death. Only for `ssam-d`.")
        parser.add_argument('--growth_strategy', type=str, default='random', help="Only for `ssam-d`.")
        return parser

    def lr_scheduler_parser(self):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--warmup_epoch', type=int, default=0)
        parser.add_argument('--warmup_init_lr', type=float, default=0.0)
        parser.add_argument('--lr_scheduler', type=str, default='CosineLRscheduler')
        # CosineLRscheduler
        parser.add_argument('--eta_min', type=float, default=0)
        # MultiStepLRscheduler
        parser.add_argument('--milestone', type=int, nargs='+', default=[60, 120, 160], help="Milestone for MultiStepLRscheduler.")
        parser.add_argument('--gamma', type=float, default=0.2, help="Gamma for MultiStepLRscheduler.")
        return parser

    def model_parser(self):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--model', type=str, default='resnet18', help="Model in registry to use.")
        return parser


    def get_args(self):
        all_parser_funcs = []
        for func_or_attr in dir(self):
            if callable(getattr(self, func_or_attr)) and not func_or_attr.startswith('_') and func_or_attr[-len('parser'):] == 'parser':
                all_parser_funcs.append(getattr(self, func_or_attr))
        all_parsers = [parser_func() for parser_func in all_parser_funcs]
        
        final_parser = argparse.ArgumentParser(parents=all_parsers)
        args = final_parser.parse_args()
        self.auto_set_name(args)
        return args

    def auto_set_name(self, args):

        def sam_hyper_param(args):
            args_opt = args.opt.split('-')
            if len(args_opt) == 1:
                return []
            elif len(args_opt) == 2:
                sam_opt, base_opt = args_opt[0], args_opt[1]
            # SAM, SSAMF, SSAMD
            output_name = ['rho{}'.format(args.rho)]
            if sam_opt[:4].upper() == 'SSAM':
                output_name.extend(['s{}u{}'.format(args.sparsity, args.update_freq), 'D{}{}'.format(args.drop_rate, args.drop_strategy), 'R{}'.format(args.growth_strategy), 'fisher-n{}'.format(args.num_samples)])
            return output_name

        if args.output_name is None:
            args.output_name = '_'.join([
                args.dataset,
                'bsz' + str(args.batch_size),
                'epoch' + str(args.epochs),
                args.model,
                'lr' + str(args.lr),
                str(args.opt),
            ] + sam_hyper_param(args) + ['seed{}'.format(args.seed)])
        if args.wandb_name == 'Default':
            args.wandb_name = args.output_name