cuda=7
model=vgg11_bn
# s=0.9
for update_freq in 1 5
do
    for num_samples in 128 1024
    do
        for seed in 1234 4321 6789 9876
        do
            CUDA_VISIBLE_DEVICES=$cuda python train.py --sparsity 0.9 --update_freq $update_freq --num_samples $num_samples --dataset CIFAR10_cutout --seed $seed --lr 0.1 --weight_decay 1e-3 --model $model --opt ssamf-sgd --rho 0.1 --wandb --output_dir logs/${model}_${data}
        done
    done
done
# s=0.95
for update_freq in 1 5
do
    for num_samples in 128 1024
    do
        for seed in 1234 4321 6789 9876
        do
            CUDA_VISIBLE_DEVICES=$cuda python train.py --sparsity 0.95 --update_freq $update_freq --num_samples $num_samples --dataset CIFAR10_cutout --seed $seed --lr 0.1 --weight_decay 1e-3 --model $model --opt ssamf-sgd --rho 0.1 --wandb --output_dir logs/${model}_${data}
        done
    done
done
# s=0.98
for update_freq in 1 5
do
    for num_samples in 128 1024
    do
        for seed in 1234 4321 6789 9876
        do
            CUDA_VISIBLE_DEVICES=$cuda python train.py --sparsity 0.98 --update_freq $update_freq --num_samples $num_samples --dataset CIFAR10_cutout --seed $seed --lr 0.1 --weight_decay 1e-3 --model $model --opt ssamf-sgd --rho 0.1 --wandb --output_dir logs/${model}_${data}
        done
    done
done
# s=0.99
for update_freq in 1 5
do
    for num_samples in 128 1024
    do
        for seed in 1234 4321 6789 9876
        do
            CUDA_VISIBLE_DEVICES=$cuda python train.py --sparsity 0.99 --update_freq $update_freq --num_samples $num_samples --dataset CIFAR10_cutout --seed $seed --lr 0.1 --weight_decay 1e-3 --model $model --opt ssamf-sgd --rho 0.1 --wandb --output_dir logs/${model}_${data}
        done
    done
done
# for seed in 1234 4321 6789 9876
# do
#     python train.py --seed $seed --dataset CIFAR100_cutout --model wideresnet28x10 --wandb --output_dir logs/${model}_${data} --opt sgd
# done

# for rho in 0.01 0.02 0.05 0.1 0.2
# do
#     for seed in 1234 4321 6789 9876
#     do
#         python train.py --seed $seed --dataset CIFAR100_cutout --weight_decay 1e-3 --model wideresnet28x10 --wandb --output_dir logs/${model}_${data} --opt sam-sgd --rho $rho
#     done
# done

# for seed in 1234 4321 6789 9876
# do
#     for data in CIFAR10_cutout CIFAR100_cutout
#     do
#         for model in resnet18 resnet34
#         do
#             python train.py --dataset $data --seed $seed --model $model --wandb --output_dir logs/${model}_${data}
#         done
#     done
# done
# cifar10 + resnet18 (rho=0.1)
# cuda=0
# for seed in 1234 4321 6789 9876
# do
#     for sparsity in 0.5 0.8 0.9 
#     do
#         for update_freq in 1 2 5 10
#         do
#             for num_samples in 128 512 1024
#             do
#                 CUDA_VISIBLE_DEVICES=$cuda python train.py --sparsity $sparsity --update_freq $update_freq --num_samples $num_samples --dataset CIFAR10_cutout --seed $seed --model resnet18 --opt ssamf-sgd --weight_decay 0.001 --rho 0.1 --wandb --output_dir logs/resnet18_CIFAR10_cutout
#             done
#         done
#     done
# done
# cuda=1
# for seed in 1234 4321 6789 9876
# do
#     for sparsity in 0.95 0.98 0.99
#     do
#         for update_freq in 1 2 5 10
#         do
#             for num_samples in 128 512 1024
#             do
#                 CUDA_VISIBLE_DEVICES=$cuda python train.py --sparsity $sparsity --update_freq $update_freq --num_samples $num_samples --dataset CIFAR10_cutout --seed $seed --model resnet18 --opt ssamf-sgd --weight_decay 0.001 --rho 0.1 --wandb --output_dir logs/resnet18_CIFAR10_cutout
#             done
#         done
#     done
# done