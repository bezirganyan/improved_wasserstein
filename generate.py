'''Attack a CIFAR-10 model with a Wasserstein adversary.'''
import time

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import os
import argparse

from torchvision import models
import torch.multiprocessing as mp
from utils import progress_bar
from pgd import attack


def craft(rank, args, dataloader_list, dataloader_path, result_path, eps_list):
    if args.norm == 'linfinity':
        args.alpha = 0.1
    elif args.norm == 'grad':
        args.alpha = 0.06
    elif args.norm == 'enhanced_linfinity':
        args.alpha = 0.04
    if args.preset == 'new_clamping':
        args.unconstrained = False
        args.no_clamping = False
    elif args.preset == 'old_clamping':
        args.unconstrained = True
        args.no_clamping = False
    elif args.preset == 'old_linf':
        args.unconstrained = True
        args.no_clamping = True
    else:
        assert False, f'Unknown preset: {args.preset}'

    fparam = dataloader_list[0].split('_')
    device = torch.device(f'cuda:{rank}')

    # Data
    print(f'{rank} ==> Preparing data..')

    testloader = torch.load(os.path.join(dataloader_path, f'{fparam[0]}_{rank}_dtl.pt'))
    torch.save(testloader, f'{result_path}/improved_{rank}_dtl.pt')
    mu = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float, device=device).unsqueeze(-1).unsqueeze(-1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float, device=device).unsqueeze(-1).unsqueeze(-1)

    unnormalize = lambda x: x*std + mu
    normalize = lambda x: (x-mu)/std

    # Model architecture is from pytorch-cifar submodule
    print(f'{rank} ==> Building model..')
    net = models.vgg16(pretrained=True)
    net = net.to(device)
    # if device != 'cpu':
    #     net = torch.nn.DataParallel(net)
    #     cudnn.benchmark = True

    regularization = args.reg
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # freeze parameters
    for p in net.parameters():
        p.requires_grad = False

    criterion = nn.CrossEntropyLoss()

    print(f'{rank} ==> regularization set to {regularization}')
    print(f'{rank} ==> p set to {regularization}')
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for eps in eps_list:
        adv_samples = []
        perturbations = []
        durations = []
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            start_time = time.time()
            inputs_pgd, _, epsilons = attack(torch.clamp(unnormalize(inputs),min=0),
                                             targets, net,
                                             normalize=normalize,
                                             regularization=regularization,
                                             p=args.p,
                                             alpha=args.alpha,
                                             norm = args.norm,
                                             ball = args.ball,
                                             epsilon_iters=args.epsilon_iters,
                                             epsilon = eps,
                                             epsilon_factor=args.epsilon_factor,
                                             clamping=not args.no_clamping,
                                             use_tqdm=True,
                                             constrained_sinkhorn=not args.unconstrained,
                                             maxiters=args.maxiters)
            duration = time.time() - start_time
            durations.append(duration)
            if (epsilons.mean() - eps).abs() > 1e-6:
                print(f'{rank} ==> epsilon is not constant! d_eps = {(epsilons.mean() - eps).abs()}')
                raise ValueError(f'{epsilons.mean()} != {eps}')

            outputs_pgd = net(normalize(inputs_pgd))
            loss = criterion(outputs_pgd, targets)

            test_loss += loss.item()
            _, predicted = outputs_pgd.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            adv_samples.append(inputs_pgd.detach().cpu())
            perturbation = inputs_pgd - inputs
            perturbations.append(perturbation.detach().cpu())

            progress_bar(batch_idx, len(testloader), '%d ==> Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (rank, test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            print('\n')

            acc = 100.*correct/total
        adv_samples = torch.cat(adv_samples, dim=0)
        perturbations = torch.cat(perturbations, dim=0)
        adv_path = f'{result_path}/improved_wasserstein_{eps:.5f}_{rank}.pt'
        prt_path = f'{result_path}/improved_wasserstein_{eps:.5f}_{rank}_prt.pt'
        dur_path = f'{result_path}/improved_wasserstein_{eps:.5f}_{rank}_dur.pt'
        torch.save(adv_samples, adv_path)
        torch.save(perturbations, prt_path)
        torch.save(durations, dur_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Attack a CIFAR-10 model with a Wasserstein adversary.')
    parser.add_argument('--model', default='resnet18')
    # Directories
    parser.add_argument('--outdir', default='epsilons/', help='output dir')
    parser.add_argument('--datadir', default='data/', help='output dir')
    # Threat model
    parser.add_argument('--norm', default='grad')
    parser.add_argument('--ball', default='wasserstein')
    parser.add_argument('--p', default=1, type=float, help='p-wasserstein distance')
    parser.add_argument('--alpha', default=0.06, type=float, help='PGD step size')
    # Sinkhorn projection
    parser.add_argument('--reg', default=3000, type=float, help='entropy regularization')
    # Attack schedule
    # parser.add_argument('--init-epsilon', default=0.001, type=float, help='initial epsilon')
    parser.add_argument('--epsilon-iters', default=1, type=int, help='freq to ramp up epsilon')
    parser.add_argument('--epsilon-factor', default=.001, type=float, help='factor to ramp up epsilon')
    parser.add_argument('--maxiters', default=400, type=int, help='PGD num of steps')
    # MISC
    parser.add_argument('--unconstrained', action='store_true')
    parser.add_argument('--no-clamping', action='store_true')
    parser.add_argument('--preset', default='new_clamping')

    parser.add_argument('--dtl_folder', type=str, default='../adversarial_arena/results')
    parser.add_argument('--result_path', type=str, default='results_iwass')
    parser.add_argument('--n_procs', type=int, default=8)
    parser.add_argument('--eps_start', type=float, default=0.05)
    parser.add_argument('--eps_end', type=float, default=1.0)
    parser.add_argument('--eps_count', type=int, default=15)

    args = parser.parse_args()

    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    file_list = os.listdir(args.dtl_folder)
    dataloader_list = [f for f in file_list if f.endswith('dtl.pt')]

    eps_list = np.linspace(args.eps_start, args.eps_end, args.eps_count)
    mp.spawn(fn=craft, args=(args, dataloader_list, args.dtl_folder, args.result_path, eps_list), nprocs=args.n_procs)