#!/usr/bin/env python3
"""
gpu_utils.py — GPU 资源动态检测与 worker 分配

运行时查询每块 GPU 的空闲显存，根据模型显存需求计算可分配的 worker 数，
尊重其他用户对 GPU 的占用。
"""

import torch


def detect_available_gpus(
    min_free_mb: int = 9000,
    mem_per_worker_mb: int = 8000,
    max_gpus: int = 0,
    max_workers_per_gpu: int = 1,
) -> list[dict]:
    """
    检测可用 GPU 并计算每块 GPU 可放置的 worker 数。

    Args:
        min_free_mb: GPU 最低空闲显存（MB），低于此值的 GPU 不使用
        mem_per_worker_mb: 每个 worker 需要的显存（MB），
                           DeepLIIF (~3GB) + SAM2 (~3GB) + 推理开销 ≈ 8GB
        max_gpus: 最多使用几块 GPU，0 表示不限制（全部可用的都用）
        max_workers_per_gpu: 每块 GPU 最多放几个 worker

    Returns:
        list[dict]: 按空闲显存降序排列
            [{gpu_id: int, free_mb: int, total_mb: int, workers: int}, ...]
            如果没有可用 GPU，返回空列表
    """
    if not torch.cuda.is_available():
        print("  No CUDA GPUs available.")
        return []

    num_gpus = torch.cuda.device_count()
    gpu_info = []

    print(f"\n  GPU Resource Detection ({num_gpus} GPUs found):")
    print(f"  {'GPU':>5} {'Name':>30} {'Total':>10} {'Free':>10} {'Workers':>8}")
    print(f"  {'-'*5} {'-'*30} {'-'*10} {'-'*10} {'-'*8}")

    for i in range(num_gpus):
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(i)
        except Exception as e:
            # GPU 完全被占满时 mem_get_info 可能抛 CUDA OOM
            name = torch.cuda.get_device_name(i)
            print(f"  {i:>5} {name:>30}    (query failed: {e})")
            continue

        free_mb = free_bytes // (1024 * 1024)
        total_mb = total_bytes // (1024 * 1024)
        name = torch.cuda.get_device_name(i)

        if free_mb >= min_free_mb:
            workers = min(free_mb // mem_per_worker_mb, max_workers_per_gpu)
            workers = max(workers, 1)  # 至少 1 个
        else:
            workers = 0

        status = f"{workers}" if workers > 0 else "skip"
        print(f"  {i:>5} {name:>30} {total_mb:>8} MB {free_mb:>8} MB {status:>8}")

        if workers > 0:
            gpu_info.append({
                'gpu_id': i,
                'free_mb': free_mb,
                'total_mb': total_mb,
                'workers': workers,
            })

    # 按空闲显存降序排列（优先用空闲最多的卡）
    gpu_info.sort(key=lambda x: x['free_mb'], reverse=True)

    # 限制最大 GPU 数量
    if max_gpus > 0:
        gpu_info = gpu_info[:max_gpus]

    total_workers = sum(g['workers'] for g in gpu_info)
    print(f"\n  Available: {len(gpu_info)} GPUs, {total_workers} total workers "
          f"(min_free={min_free_mb}MB, per_worker={mem_per_worker_mb}MB)")

    return gpu_info


def build_worker_assignments(
    gpu_info: list[dict],
    num_tiles: int,
) -> list[dict]:
    """
    将 tiles 均匀分配给各 worker。

    Args:
        gpu_info: detect_available_gpus() 返回的 GPU 列表
        num_tiles: 需要处理的 tile 总数

    Returns:
        list[dict]: 每个 worker 的分配信息
            [{gpu_id: int, worker_id: int, tile_start: int, tile_end: int}, ...]
    """
    total_workers = sum(g['workers'] for g in gpu_info)
    if total_workers == 0:
        return []

    # 计算每个 worker 分到多少 tiles
    base_count = num_tiles // total_workers
    remainder = num_tiles % total_workers

    assignments = []
    tile_offset = 0
    worker_id = 0

    for gpu in gpu_info:
        for _ in range(gpu['workers']):
            # 前 remainder 个 worker 各多分 1 个 tile
            count = base_count + (1 if worker_id < remainder else 0)
            if count > 0:
                assignments.append({
                    'gpu_id': gpu['gpu_id'],
                    'worker_id': worker_id,
                    'tile_start': tile_offset,
                    'tile_end': tile_offset + count,
                })
            tile_offset += count
            worker_id += 1

    return assignments
