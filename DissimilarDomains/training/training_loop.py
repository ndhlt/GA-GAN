﻿# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import time
import copy
import json
import pickle
import psutil
import PIL.Image
import numpy as np
import regex
import torch
import torch.utils.data
import torch.backends.cuda
import torch.backends.cudnn
import dnnlib
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix
import legacy
from metrics import metric_main


# ----------------------------------------------------------------------------

def set_requires_grad_by_filter(module, param_filter, verbose):
    for name, param in module.named_parameters():
        if param_filter(name):
            param.requires_grad_(True)
            if verbose:
                print('{0: <60}: {1: <30} {2:d}'.format(name, str(list(param.shape)), param.numel()))


def check_block_resolution(pname, resolution):
    if resolution is None:
        return 'synthesis' in pname
    return f'synthesis.b{resolution}' in pname


def log_histograms(writer, module, step, requires_grad):
    for name, param in module.named_parameters():
        if requires_grad and not param.requires_grad:
            continue
        if len(param.shape) == 0:
            pass
        else:
            writer.add_histogram(f'{name}', param.flatten(), global_step=step, bins='tensorflow')


name_filters = {
        'mapping': lambda res: lambda pname: 'mapping' in pname,

        'tRGB_affine': lambda res: lambda pname: check_block_resolution(pname, res) and 'torgb.affine' in pname,
        'tRGB_conv': lambda res: lambda pname: check_block_resolution(pname, res) and ('torgb.weight' in pname or 'torgb.bias' in pname) and 'affine' not in pname and 'offset' not in pname,
        'tRGB_offset': lambda res: lambda pname: check_block_resolution(pname, res) and ('torgb.offset' in pname and 'torgb.weights_offset' not in pname),
        'tRGB_weights_offset': lambda res: lambda pname: check_block_resolution(pname, res) and 'torgb.weights_offset' in pname,
        'tRGB_affine_weights_offset': lambda res: lambda pname: check_block_resolution(pname, res) and 'torgb.affine.weights_offset' in pname,

        'synt_affine': lambda res: lambda pname: check_block_resolution(pname, res) and 'conv' in pname and 'affine' in pname,
        'synt_conv': lambda res: lambda pname: check_block_resolution(pname, res) and 'conv' in pname and ('weight' in pname or 'noise_strength' in pname or 'bias' in pname) and 'affine' not in pname and 'offset' not in pname,
        'synt_const': lambda res: lambda pname: check_block_resolution(pname, res) and 'const' in pname,
        'synt_offset': lambda res: lambda pname: check_block_resolution(pname, res) and 'conv' in pname and ('offset' in pname and 'weights_offset' not in pname),
        'synt_weights_offset': lambda res: lambda pname: check_block_resolution(pname, res) and 'conv' in pname and 'affine' not in pname and 'weights_offset' in pname,
        'synt_affine_weights_offset': lambda res: lambda pname: check_block_resolution(pname, res) and 'conv' in pname and 'affine.weights_offset' in pname
}


def set_requires_grad(module, parts, verbose=False):
    """
    Args:
        module: trainable module
        parts: list of parameters groups for which requires_grad will be set
        verbose:
    Returns:

    """
    assert isinstance(parts, list)

    module.requires_grad_(False)
    if 'all' in parts:
        module.requires_grad_(True)
        return

    for name, name_filter in name_filters.items():
        for resolution in [1024, 512, 256, 128, 64, 32, 16, 8, 4, None]:
            target_name = name if resolution is None else f'{name}.b{resolution}'
            if target_name in parts:
                set_requires_grad_by_filter(module, name_filter(resolution), verbose=verbose)

    # ----------------------------------------------------------------------------


def setup_snapshot_image_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(7680 // training_set.image_shape[2], 7, 32)
    gh = np.clip(4320 // training_set.image_shape[1], 4, 32)

    # No labels => show random subset of training samples.
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        label_groups = dict()  # label => [idx, ...]
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data.
    images, labels = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)


# ----------------------------------------------------------------------------

def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    # noinspection PyPep8Naming
    _N, C, H, W = img.shape
    img = img.reshape(gh, gw, C, H, W)
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape(gh * H, gw * W, C)

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)


# ----------------------------------------------------------------------------

# noinspection PyDefaultArgument,PyPep8Naming
def training_loop(
        run_dir='.',  # Output directory.
        training_set_kwargs={},  # Options for training set.
        data_loader_kwargs={},  # Options for torch.utils.data.DataLoader.
        G_kwargs={},  # Options for generator network.
        D_kwargs={},  # Options for discriminator network.
        G_opt_kwargs={},  # Options for generator optimizer.
        D_opt_kwargs={},  # Options for discriminator optimizer.
        augment_kwargs=None,  # Options for augmentation pipeline. None = disable.
        loss_kwargs={},  # Options for loss function.
        metrics=[],  # Metrics to evaluate during training.
        random_seed=0,  # Global random seed.
        num_gpus=1,  # Number of GPUs participating in the training.
        rank=0,  # Rank of the current process in [0, num_gpus[.
        batch_size=4,  # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
        batch_gpu=4,  # Number of samples processed at a time by one GPU.
        ema_kimg=10,  # Half-life of the exponential moving average (EMA) of generator weights.
        ema_rampup=None,  # EMA ramp-up coefficient.
        G_reg_interval=4,  # How often to perform regularization for G? None = disable lazy regularization.
        D_reg_interval=16,  # How often to perform regularization for D? None = disable lazy regularization.
        augment_p=0,  # Initial value of augmentation probability.
        ada_target=None,  # ADA target value. None = fixed p.
        ada_interval=4,  # How often to perform ADA adjustment?
        # ADA adjustment speed, measured in how many kimg it takes for p to increase/decrease by one unit.
        ada_kimg=500,
        total_kimg=25000,  # Total length of the training, measured in thousands of real images.
        kimg_per_tick=4,  # Progress snapshot interval.
        image_snapshot_ticks=50,  # How often to save image snapshots? None = disable.
        network_snapshot_ticks=50,  # How often to save network snapshots? None = disable.
        resume_pkl=None,  # Network pickle to resume training from.
        cudnn_benchmark=True,  # Enable torch.backends.cudnn.benchmark?
        allow_tf32=False,  # Enable torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32?
        # Callback function for determining whether to abort training. Must return consistent results across ranks.
        abort_fn=None,
        progress_fn=None,  # Callback function for updating training progress. Called for all ranks.
        requires_grad_parts={},
        threshold=0.5, #GA rated scores
):
    # Initialize.
    start_time = time.time()
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark  # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32  # Allow PyTorch to internally use tf32 for matmul
    torch.backends.cudnn.allow_tf32 = allow_tf32  # Allow PyTorch to internally use tf32 for convolutions
    conv2d_gradfix.enabled = True  # Improves training speed.
    grid_sample_gradfix.enabled = True  # Avoids errors with the augmentation pipe.

    # Load training set.
    if rank == 0:
        print('Loading training set...')
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs)  # subclass of training.dataset.Dataset
    training_set_sampler = misc.InfiniteSampler(
        dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed
    )
    training_set_iterator = iter(torch.utils.data.DataLoader(
        dataset=training_set, sampler=training_set_sampler, batch_size=batch_size // num_gpus, **data_loader_kwargs
    ))
    if rank == 0:
        print()
        print('Num images: ', len(training_set))
        print('Image shape:', training_set.image_shape)
        print('Label shape:', training_set.label_shape)
        np.savetxt(os.path.join(run_dir, f'{len(training_set)}-shot-index.txt'), training_set._raw_idx)
        print()

    # Construct networks.
    if rank == 0:
        print('Constructing networks...')
    common_kwargs = dict(
        c_dim=training_set.label_dim,
        img_resolution=training_set.resolution,
        img_channels=training_set.num_channels
    )
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device)  # subclass of torch.nn.Module
    D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(device)  # subclass of torch.nn.Module
    G_ema = copy.deepcopy(G).eval()

    # Resume from existing pickle.
    if (resume_pkl is not None) and (rank == 0):
        print(f'Resuming from "{resume_pkl}"')
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
        for name, module in [('G', G), ('D', D), ('G_ema', G_ema)]:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)

    # Print network summary tables.
    if rank == 0:
        z = torch.empty([batch_gpu, G.z_dim], device=device)
        c = torch.empty([batch_gpu, G.c_dim], device=device)
        img = misc.print_module_summary(G, [z, c])
        misc.print_module_summary(D, [img, c])

    # Setup augmentation.
    if rank == 0:
        print('Setting up augmentation...')
    augment_pipe = None
    ada_stats = None
    if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None):
        augment_pipe = dnnlib.util.construct_class_by_name(
            **augment_kwargs
        ).train().requires_grad_(False).to(device)  # subclass of torch.nn.Module
        augment_pipe.p.copy_(torch.as_tensor(augment_p))
        if ada_target is not None:
            ada_stats = training_stats.Collector(regex='Loss/signs/real')

    # Distribute across GPUs.
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    ddp_modules = dict()
    for name, module in [
        ('G_mapping', G.mapping),
        ('G_synthesis', G.synthesis),
        ('D', D), (None, G_ema),
        ('augment_pipe', augment_pipe)
    ]:
        if (num_gpus > 1) and (module is not None) and len(list(module.parameters())) != 0:
            module.requires_grad_(True)
            module = torch.nn.parallel.DistributedDataParallel(module, device_ids=[device], broadcast_buffers=False)
            module.requires_grad_(False)
        if name is not None:
            ddp_modules[name] = module

    # Setup training phases.
    if rank == 0:
        print('Setting up training phases...')
    loss = dnnlib.util.construct_class_by_name(
        device=device, **ddp_modules, **loss_kwargs
    )  # subclass of training.loss.Loss
    phases = []
    for name, module, opt_kwargs, reg_interval in [
        ('G', G, G_opt_kwargs, G_reg_interval),
        ('D', D, D_opt_kwargs, D_reg_interval)
    ]:
        if reg_interval is None:
            opt = dnnlib.util.construct_class_by_name(
                params=module.parameters(), **opt_kwargs
            )  # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name + 'both', module=module, opt=opt, interval=1)]
        else:  # Lazy regularization.
            mb_ratio = reg_interval / (reg_interval + 1)
            opt_kwargs = dnnlib.EasyDict(opt_kwargs)
            opt_kwargs.lr = opt_kwargs.lr * mb_ratio
            opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
            opt = dnnlib.util.construct_class_by_name(
                module.parameters(), **opt_kwargs
            )  # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name + 'main', module=module, opt=opt, interval=1)]
            phases += [dnnlib.EasyDict(name=name + 'reg', module=module, opt=opt, interval=reg_interval)]
    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    # Export sample images.
    grid_size = None
    grid_z = None
    grid_c = None
    if rank == 0:
        print('Exporting sample images...')
        grid_size, images, labels = setup_snapshot_image_grid(training_set=training_set)
        save_image_grid(images, os.path.join(run_dir, 'reals.png'), drange=[0, 255], grid_size=grid_size)
        grid_z = torch.randn([labels.shape[0], G.z_dim], device=device).split(batch_gpu)
        grid_c = torch.from_numpy(labels).to(device).split(batch_gpu)
        images = torch.cat([G_ema(z=z, c=c, noise_mode='const').cpu() for z, c in zip(grid_z, grid_c)]).numpy()
        save_image_grid(images, os.path.join(run_dir, 'fakes_init.png'), drange=[-1, 1], grid_size=grid_size)

    # Initialize logs.
    if rank == 0:
        print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print('Skipping tfevents export:', err)

    if rank == 0:
        print('Exporting model graph...')

        # noinspection PyPep8Naming,PyShadowingNames
        class _WrapperModel(torch.nn.Module):
            def __init__(self, G, D):
                super().__init__()
                self.G = G
                self.D = D

            def forward(self, z, c):
                return self.D(self.G(z, c), c)

        graph_z = torch.randn([batch_gpu, G.z_dim], device=device)
        graph_c = torch.from_numpy(np.empty([batch_gpu, 0])).to(device)
        stats_tfevents.add_graph(_WrapperModel(G, D), [graph_z, graph_c])

    if rank == 0:
        print('List requires_grad parameters...')
        set_requires_grad(G, parts=requires_grad_parts['Gboth'], verbose=True)
        set_requires_grad(D, parts=requires_grad_parts['Dboth'], verbose=True)

        G.requires_grad_(False)
        D.requires_grad_(False)

    if rank == 0:
        print('Exporting histograms...')

        log_histograms(stats_tfevents, G, step=-kimg_per_tick, requires_grad=False)
        log_histograms(stats_tfevents, D, step=-kimg_per_tick, requires_grad=False)

    # Train.
    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        print()
    cur_nimg = 0
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0
    if progress_fn is not None:
        progress_fn(0, total_kimg)

# GAの適用部分をここに挿入

    def compute_gradient_penalty(D, real_samples, fake_samples, device):
        alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = D(interpolates)
        fake = torch.ones(d_interpolates.size(), requires_grad=False, device=device)
        gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,

        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def apply_genetic_algorithm(G, D, D_real, D_fake, phase_real_img, phase_gen_img, device, threshold=0.5):
        # 似ている画像を抽出
        similar_imgs_mask = (torch.abs(D_real - D_fake) < threshold)

        if similar_imgs_mask.sum() == 0:
            return phase_gen_img
    
        similar_real_imgs = phase_real_img[similar_imgs_mask]
        similar_fake_imgs = phase_gen_img[similar_imgs_mask]

        # 特徴抽出
        real_features = extract_features(D, similar_real_imgs)
        fake_features = extract_features(D, similar_fake_imgs)

        # 遺伝的アルゴリズムを適用
        crossover_features = gaussian_crossover(real_features, fake_features)
        mutated_features = dynamic_mutation(crossover_features)

        # 新たな画像を生成
        new_generated_imgs = G(mutated_features)
        phase_gen_img[similar_imgs_mask] = new_generated_imgs
    
        return phase_gen_img
    
    while True:

        # Fetch training data.
        with torch.autograd.profiler.record_function('data_fetch'):
            phase_real_img, phase_real_c = next(training_set_iterator)
            phase_real_img = (phase_real_img.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
            phase_real_c = phase_real_c.to(device).split(batch_gpu)
            all_gen_z = torch.randn([len(phases) * batch_size, G.z_dim], device=device)
            all_gen_z = all_gen_z.split(batch_size)  # 分割する
            # 修正部分: all_gen_zをさらにbatch_gpu単位で分割する
            all_gen_z = [z.split(batch_gpu) for z in all_gen_z]  # さらにbatch_gpuごとに分割

         # 修正部分: all_gen_cをbatch_size単位で分割する
            all_gen_c = [
                training_set.get_label(np.random.randint(len(training_set)))
                for _ in range(len(phases) * batch_size)
            ]
            all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)
            all_gen_c = all_gen_c.split(batch_size)  # 分割する
            # 修正部分: all_gen_cをさらにbatch_gpu単位で分割する
            all_gen_c = [c.split(batch_gpu) for c in all_gen_c]  # さらにbatch_gpuごとに分割
    

        # Execute training phases.
        for phase, phase_gen_z, phase_gen_c in zip(phases, all_gen_z, all_gen_c):
            if batch_idx % phase.interval != 0:
                continue

            # Initialize gradient accumulation.
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))
            phase.opt.zero_grad(set_to_none=True)
            set_requires_grad(phase.module, parts=requires_grad_parts[phase.name])

       # Gを使用してフェーズごとに生成画像を作成
        for z, c in zip(phase_gen_z, phase_gen_c):
            phase_gen_img = G(z, c)  # 各 z, c ペアごとに生成

        D_real_output = D(phase_real_img, phase_real_c)
        D_fake_output = D(phase_gen_img, c)

        if isinstance(D_real_output, tuple):
            D_real_output = D_real_output[0]  # タプルの最初の要素を使用
        if isinstance(D_fake_output, tuple):
            D_fake_output = D_fake_output[0]  # タプルの最初の要素を使用

        # D_real_outputやD_fake_outputがタプルの場合、それぞれ最初の要素を使用
        if isinstance(D_real_output, tuple):
            D_real_output = D_real_output[0]
        if isinstance(D_fake_output, tuple):
            D_fake_output = D_fake_output[0]

        # detach() メソッドを呼び出して、D_real と D_fake を更新
        D_real = D_real_output.detach()
        D_fake = D_fake_output.detach()

        # GAシステムを適用
        phase_gen_img = apply_genetic_algorithm(G, D, D_real_output, D_fake_output, phase_real_img, phase_gen_img, device, threshold)    # 他の処理が必要であれば、ここに追加
            # Accumulate gradients over multiple rounds.
        for round_idx, (real_img, real_c, gen_z, gen_c) in enumerate(
            zip(phase_real_img, phase_real_c, phase_gen_z, phase_gen_c)
            ):
            sync = (round_idx == batch_size // (batch_gpu * num_gpus) - 1)
            gain = phase.interval
            loss.accumulate_gradients(
                phase=phase.name, real_img=real_img, real_c=real_c, gen_z=gen_z, gen_c=gen_c, sync=sync, gain=gain
            )

            # Update weights.
    phase.module.requires_grad_(False)
    with torch.autograd.profiler.record_function(phase.name + '_opt'):
        for param in phase.module.parameters():
            if param.grad is not None:
                misc.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
                phase.opt.step()
    if phase.end_event is not None:
        phase.end_event.record(torch.cuda.current_stream(device))

        # Update G_ema.
        with torch.autograd.profiler.record_function('Gema'):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        # Execute ADA heuristic.
        if (ada_stats is not None) and (batch_idx % ada_interval == 0):
            ada_stats.update()
            adjust = np.sign(
                ada_stats['Loss/signs/real'] - ada_target
            ) * (batch_size * ada_interval) / (ada_kimg * 1000)
            augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
    
    if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
        cur_tick += 1

        # Print status line, accumulating the same information in stats_collector.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [
            f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"
        ]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [
            f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"
        ]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [
            f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2 ** 30):<6.2f}"
        ]
        fields += [
            f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2 ** 30):<6.2f}"
        ]
        torch.cuda.reset_peak_memory_stats()
        fields += [
            f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"
        ]
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        if rank == 0:
            print(' '.join(fields))

        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print('Aborting...')

        # Save image snapshot.
        if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            images = torch.cat([G_ema(z=z, c=c, noise_mode='const').cpu() for z, c in zip(grid_z, grid_c)]).numpy()
            save_image_grid(
                images, os.path.join(run_dir, f'fakes{cur_nimg // 1000:06d}.png'),
                drange=[-1, 1], grid_size=grid_size
            )

        # Save network snapshot.
        snapshot_pkl = None
        snapshot_data = None
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs))
            for name, module in [('G', G), ('D', D), ('G_ema', G_ema), ('augment_pipe', augment_pipe)]:
                if module is not None:
                    if num_gpus > 1:
                        misc.check_ddp_consistency(module, ignore_regex=r'.*\.w_avg')
                    module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
                snapshot_data[name] = module
                del module  # conserve memory
            snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg // 1000:06d}.pkl')
            if rank == 0:
                with open(snapshot_pkl, 'wb') as f:
                    pickle.dump(snapshot_data, f)

        # Evaluate metrics.
        if (snapshot_data is not None) and (len(metrics) > 0):
            if rank == 0:
                print('Evaluating metrics...')
            for metric in metrics:
                result_dict = metric_main.calc_metric(
                    metric=metric, G=snapshot_data['G_ema'],
                    dataset_kwargs=training_set_kwargs, num_gpus=num_gpus, rank=rank,
                    device=device
                )
                if rank == 0:
                    metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
                stats_metrics.update(result_dict.results)
        del snapshot_data  # conserve memory

        # Collect statistics.
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('Timing/' + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs.
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)

            set_requires_grad(G, parts=requires_grad_parts['Gboth'])
            set_requires_grad(D, parts=requires_grad_parts['Dboth'])

            log_histograms(stats_tfevents, G, step=global_step, requires_grad=True)
            log_histograms(stats_tfevents, D, step=global_step, requires_grad=True)

            G.requires_grad_(False)
            D.requires_grad_(False)

            stats_tfevents.flush()
        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            pritn("loop end")

    # Done.
    if rank == 0:
        print()
        print('Exiting...')

# ----------------------------------------------------------------------------
