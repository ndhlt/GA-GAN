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
    from GA.feature_extraction import extract_features
    from GA.crossover_mutation import gaussian_crossover, dynamic_mutation
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
    'mapping': lambda res: 'mapping' in res,
    'synthesis': lambda res: 'synthesis' in res,
    'conv': lambda res: 'conv' in res,
    'torgb': lambda res: 'torgb' in res,
    }
    # ----------------------------------------------------------------------------
    def set_requires_grad(module, parts, verbose=True):
    for part_name, part_func in parts.items():
    if part_func is None:
    continue
    part_requires_grad = part_func(module)
    if part_requires_grad:
    set_requires_grad_by_filter(module, lambda x: name_filters[part_name](x), verbose)
    def init_writers(log_dir, rank):
    writers = {}
    if rank == 0:
    writers['log_dir'] = os.path.join(log_dir, 'events')
    writers['stats_tfevents'] = training_stats.init_writers(writers['log_dir'])
    return writers
    def train_loop(G, D, G_ema, G_opt, D_opt, training_set, training_set_iterator, num_gpus, rank, batch_size, start_time,
    loss, phases, total_kimg, kimg_per_tick, resume_pkl, cur_nimg, cur_tick, resume_kimg, ema_rampup, 
    progress_fn, augment_pipe, ada_kimg, ada_tgt, ada_interval, run_dir, log_dir):
    if rank == 0:
    print('Exporting histograms...')
    log_histograms(stats_tfevents, G, step=-kimg_per_tick, requires_grad=False)
    log_histograms(stats_tfevents, D, step=-kimg_per_tick, requires_grad=False)
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
    while True:
    # Fetch training data.
    with torch.autograd.profiler.record_function('data_fetch'):
    phase_real_img, phase_real_c = next(training_set_iterator)
    phase_real_img = (phase_real_img.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
    phase_real_c = phase_real_c.to(device).split(batch_gpu)
    all_gen_z = torch.randn([len(phases) * batch_size, G.z_dim], device=device)
    all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]
    all_gen_c = [
    training_set.get_label(np.random.randint(len(training_set)))
    for _ in range(len(phases) * batch_size)
    ]
    all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)
    all_gen_c = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen_c.split(batch_size)]
    # GAの適用部分をここに挿入
    generated_images = G(all_gen_z, all_gen_c)  # 生成画像
    generated_scores = D(generated_images)  # 識別器によるスコア
    if generated_scores.mean() > threshold:
    real_features = extract_features(D, phase_real_img)
    generated_features = extract_features(D, generated_images)
    # 交叉と突然変異の適用
    new_features = gaussian_crossover(real_features, generated_features)
    mutated_features = dynamic_mutation(new_features)
    # 再生成
    generated_images = G(mutated_features)
    # Execute training phases.
    for phase, phase_gen_z, phase_gen_c in zip(phases, all_gen_z, all_gen_c):
    if batch_idx % phase.interval != 0:
    continue
    # Initialize gradient accumulation.
    if phase.start_event is not None:
    phase.start_event.record(torch.cuda.current_stream(device))
    phase.opt.zero_grad(set_to_none=True)
    set_requires_grad(phase.module, parts=requires_grad_parts[phase.name], verbose=phase.verbose)
    # Accumulate gradients over multiple rounds.
    for round_idx, (real_img, real_c, gen_z, gen_c) in enumerate(
    zip(phase_real_img, phase_real_c, phase_gen_z, phase_gen_c)):
    sync = (round_idx == batch_size // (batch_gpu * num_gpus))
    gain = phase.interval
    loss.accumulate_gradients(
    phase=phase.name, real_img=real_img, real_c=real_c,
    gen_z=gen_z, gen_c=gen_c, sync=sync, gain=gain)
    # Update weights.
    phase.opt.step()
    # Update G_ema.
    with torch.no_grad():
    for p_ema, p in zip(G_ema.parameters(), G.parameters()):
    p_ema.copy_(p.lerp(p_ema, ema_rampup))
    # Update stats.
    stats_tfevents = training_stats.collect_stats()
    if stats_tfevents is not None:
    for stat, value in stats_tfevents.items():
    if isinstance(value, float):
    stats_tfevents[stat] = np.float32(value)
    # Perform maintenance tasks once per tick.
    done = (cur_nimg >= total_kimg * 1000)
    if (batch_idx % kimg_per_tick == 0) or done:
    tick_end_time = time.time()
    maintenance_time += tick_end_time - tick_start_time
    tick_kimg = (cur_nimg - tick_start_nimg) / 1000.0
