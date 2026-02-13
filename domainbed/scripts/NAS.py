# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse
import collections
import json
import os
import random
import sys
import time
import uuid

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
import copy

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from domainbed import model_selection
from domainbed.lib.query import Q

import optuna



def objective(trial, args, dataset, hparams):

    # 定义搜索空间
    # fusion_method = trial.suggest_categorical('fusion_method', ['concatenation', 'weighted', 'attention', 'gating'])
    # fusion_method = 'attention'
    # mlp_hidden_units = trial.suggest_categorical('mlp_hidden_units', [128, 256, 512])
    # lr = trial.suggest_categorical('learning_date', [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001])
    # mlp_hidden_layers = trial.suggest_int('mlp_hidden_layers', 1, 3)
    # activation_function = trial.suggest_categorical('activation_function', ['ReLU', 'LeakyReLU', 'Tanh'])
    info_weight = trial.suggest_uniform('fusion_weight_init', 0.0, 2.0)
    
    # 对加权融合方法设定初始权重
    # if fusion_method == 'weighted':
    #     fusion_weight_init = trial.suggest_uniform('fusion_weight_init', 0.1, 0.9)
    # else:
    #     fusion_weight_init = None
    
    # 设置模型超参数
    hparams.update({
        # 'fusion_method': fusion_method,
        # 'mlp_hidden_units': mlp_hidden_units,
        # 'activation_function': activation_function,
        # 'fusion_weight_init': fusion_weight_init,
        # 'lr': lr
        'info_weight': info_weight,
    })

    # define algorithm
    hparams['classes'] = dataset[0].classes
    hparams['device'] = device
    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
                                len(dataset) - len(args.test_envs), hparams)

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)

    # 使用少量的数据进行快速训练
    for step in range(start_step, n_steps):
        step_start_time = time.time()
        # minibatches_device = [(batch["x"].to(device), batch["y"].to(device), batch["x_g0"].to(device), batch["x_g1"].to(device),
        #                        batch["x_g2"].to(device), batch["x_g3"].to(device), batch["x_g4"].to(device))
        #                       for batch in next(train_minibatches_iterator)]
        batch = next(train_minibatches_iterator)
        minibatches_device = (batch["x"].to(device), batch["y"].to(device), batch["text"])
        # minibatches_device = (batch["x"].to(device), batch["y"].to(device), batch["x_g0"].to(device), batch["x_g1"].to(device),
        #                        batch["x_g2"].to(device), batch["x_g3"].to(device))
        if args.task == "domain_adaptation":
            uda_device = [(x.to(device), x_g.to(device))
                          for x, _, x_g, _ in next(uda_minibatches_iterator)]
        else:
            uda_device = None
        step_vals = algorithm.update(minibatches_device, uda_device)

        if algorithm.lr_scheduler is not None and step % lr_update_seq == 0: 
            algorithm.update_lr()

    # 验证模型性能
    results = {
        'step': step,
        'epoch': step / steps_per_epoch,
    }
    evals = zip(eval_loader_names, eval_loaders, eval_weights)
    temp_acc = 0
    temp_count = 0
    for name, loader, weights in evals:
        acc = misc.accuracy_clip(algorithm, loader, weights, device)
        if args.save_best_model:
            # if int(name[3]) not in args.test_envs and "out" in name:
            if int(name[3]) in args.test_envs and "in" in name:
                temp_acc += acc
                temp_count += 1
        results[name + '_acc'] = acc
    val_acc = temp_acc / (temp_count * 1.0)
    
    return val_acc


def run_optimization(args, dataset, hparams):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, args, dataset, hparams), n_trials=100)
    print("Best hyperparameters:", study.best_params)
    print("Best validation accuracy:", study.best_value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str, default="./domainbed/sub_data")
    parser.add_argument('--gen_data_dir', type=str, default="./domainbed/gen_data")
    parser.add_argument('--train_gen_dir', type=str, default="./domainbed/train_gen_data")
    parser.add_argument('--text_dir', type=str, default="./domainbed/pseudo_label/office_home_txt_clip_b")
    parser.add_argument('--dataset', type=str, default="OfficeHome")
    parser.add_argument('--algorithm', type=str, default="CLIP_Causal2")
    parser.add_argument('--task', type=str, default="domain_generalization",
                        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
                        help='JSON-serialized hparams dict', default="{\"backbone\":\"CLIP_ViT_all\",\"batch_size\":128,\"K\":20,\"lr\":5e-04,\"momentum\":0.9,\"resnet_dropout\":0.0,\"weight_decay\":0.0}")
    parser.add_argument('--hparams_seed', type=int, default=0,
                        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=1,
                        help='Trial number (used for seeding split_dataset and '
                             'random_hparams).')
    parser.add_argument('--seed', type=int, default=1,
                        help='Seed for everything else')
    # parser.add_argument('--K', type=int, default=5, help='Number of sampled images')
    parser.add_argument('--steps', type=int, default=160,
                        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
                        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="./domainbed/OfficeHome_Output/NAS/")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
                        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--save_best_model', action='store_true')
    args = parser.parse_args()
    args.save_best_model = True
    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
                                                  misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        js = json.loads(args.hparams)
        js["test_env"] = args.test_envs
        # print(args.hparams)
        hparams.update(js)

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print('device:', device)

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
                                               args.test_envs, hparams, args.gen_data_dir, args.train_gen_dir, args.text_dir)
    else:
        raise NotImplementedError
    
    in_splits = []
    out_splits = []
    uda_splits = []
    for env_i, env in enumerate(dataset):  # env is a domain
        uda = []

        out, in_ = misc.split_dataset(env,
                                      int(len(env) * args.holdout_fraction),
                                      misc.seed_hash(args.trial_seed, env_i))

        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(in_,
                                          int(len(in_) * args.uda_holdout_fraction),
                                          misc.seed_hash(args.trial_seed, env_i))

        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        if len(uda):
            uda_splits.append((uda, uda_weights))

    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")

    train_datasets = torch.utils.data.ConcatDataset([env for i, (env, env_weights) in enumerate(in_splits) if i not in args.test_envs])
    train_loaders = InfiniteDataLoader(
        dataset=train_datasets,
        weights=None,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS
    )

    uda_datasets = [env for i, (env, env_weights) in enumerate(uda_splits) if i in args.test_envs]
    if len(uda_datasets) != 0:
        uda_datasets = torch.utils.data.ConcatDataset(uda_datasets)
        uda_loaders = InfiniteDataLoader(
            dataset=uda_datasets,
            weights=None,
            batch_size=hparams['batch_size'],
            num_workers=dataset.N_WORKERS
        )
    else:
        uda_loaders = None

    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits + uda_splits)]


    eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    eval_loader_names = ['env{}_in'.format(i)
                         for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
                          for i in range(len(out_splits))]
    eval_loader_names += ['env{}_uda'.format(i)
                          for i in range(len(uda_splits))]

    train_minibatches_iterator = iter(train_loaders)
    uda_minibatches_iterator = iter(uda_loaders) if uda_loaders is not None else None

    checkpoint_vals = collections.defaultdict(lambda: [])

    # steps_per_epoch = min([len(env) / hparams['batch_size'] for env, _ in in_splits])
    steps_per_epoch = len(train_datasets) / hparams['batch_size']

    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ
    lr_update_seq = int(steps_per_epoch+0.5)

    run_optimization(args, dataset, hparams)