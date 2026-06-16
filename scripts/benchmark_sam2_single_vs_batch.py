#!/usr/bin/env python3
"""
benchmark_sam2_single_vs_batch.py — SAM2 真实 Pipeline 模拟 Benchmark

模拟真实 pipeline：N 张图像，每张图上有 M 个 cluster，
分别用 sequential / batch_size=10 / batch_size=32 三种方式对比。

每张图都会调用 set_image（图像编码），然后对该图上的所有 cluster 做 predict。

Usage:
    python scripts/benchmark_sam2_single_vs_batch.py
    python scripts/benchmark_sam2_single_vs_batch.py --num-images 50 --clusters-per-image 20
"""

import os
import sys
import time
import argparse
import json

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cd34_pipeline.sam2_wrapper.model_loader import load_sam2


def generate_random_images(num_images: int, image_size: int = 512) -> list:
    """生成 num_images 张随机 RGB 图像。"""
    np.random.seed(0)
    return [np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
            for _ in range(num_images)]


def generate_random_mask_prompts(num_prompts: int, target_size: int = 256) -> np.ndarray:
    """生成 num_prompts 个随机 mask prompt (logits 格式)。"""
    masks = np.full((num_prompts, 1, target_size, target_size), -10.0, dtype=np.float32)
    for i in range(num_prompts):
        cy = np.random.randint(30, target_size - 30)
        cx = np.random.randint(30, target_size - 30)
        radius = np.random.randint(5, 25)
        yy, xx = np.ogrid[:target_size, :target_size]
        circle = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
        masks[i, 0, circle] = 10.0
    return masks


def run_pipeline_sequential(predictor, images: list, prompts_per_image: list) -> dict:
    """
    模拟 pipeline：逐张 set_image + 逐个 predict。

    Returns:
        dict: 每张图的 set_image 时间、predict 时间、总时间
    """
    per_image_stats = []

    total_start = time.perf_counter()
    for img_idx, (image, prompts) in enumerate(zip(images, prompts_per_image)):
        # set_image
        t0 = time.perf_counter()
        predictor.set_image(image)
        torch.cuda.synchronize()
        t_set = time.perf_counter() - t0

        # 逐个 predict
        t1 = time.perf_counter()
        for j in range(prompts.shape[0]):
            mask_input = prompts[j]  # (1, 256, 256)
            masks, scores, _ = predictor.predict(
                mask_input=mask_input,
                multimask_output=True,
            )
        torch.cuda.synchronize()
        t_predict = time.perf_counter() - t1

        per_image_stats.append({
            'image_idx': img_idx,
            'set_image_ms': t_set * 1000,
            'predict_ms': t_predict * 1000,
            'total_ms': (t_set + t_predict) * 1000,
            'num_clusters': prompts.shape[0],
        })

    total_time = time.perf_counter() - total_start
    return {'total_time': total_time, 'per_image': per_image_stats}


def run_pipeline_batch(predictor, images: list, prompts_per_image: list,
                       batch_size: int = 10) -> dict:
    """
    模拟 pipeline：逐张 set_image + batch predict。

    Returns:
        dict: 每张图的 set_image 时间、predict 时间、总时间
    """
    per_image_stats = []

    total_start = time.perf_counter()
    for img_idx, (image, prompts) in enumerate(zip(images, prompts_per_image)):
        # set_image
        t0 = time.perf_counter()
        predictor.set_image(image)
        torch.cuda.synchronize()
        t_set = time.perf_counter() - t0

        # batch predict
        num_prompts = prompts.shape[0]
        t1 = time.perf_counter()

        for batch_start in range(0, num_prompts, batch_size):
            batch_end = min(batch_start + batch_size, num_prompts)
            batch_masks = prompts[batch_start:batch_end]  # (B, 1, 256, 256)

            mask_tensor = torch.as_tensor(
                batch_masks, dtype=torch.float, device=predictor.device
            )

            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                points=None, boxes=None, masks=mask_tensor,
            )

            high_res_features = [
                feat_level[-1].unsqueeze(0)
                for feat_level in predictor._features["high_res_feats"]
            ]
            low_res_masks, iou_predictions, _, _ = predictor.model.sam_mask_decoder(
                image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
                repeat_image=True,
                high_res_features=high_res_features,
            )

            all_masks = predictor._transforms.postprocess_masks(
                low_res_masks, predictor._orig_hw[-1]
            )
            all_masks = all_masks > predictor.mask_threshold

        torch.cuda.synchronize()
        t_predict = time.perf_counter() - t1

        per_image_stats.append({
            'image_idx': img_idx,
            'set_image_ms': t_set * 1000,
            'predict_ms': t_predict * 1000,
            'total_ms': (t_set + t_predict) * 1000,
            'num_clusters': num_prompts,
        })

    total_time = time.perf_counter() - total_start
    return {'total_time': total_time, 'per_image': per_image_stats}


def print_stats(label: str, result: dict):
    """打印单种方法的统计信息。"""
    stats = result['per_image']
    set_times = [s['set_image_ms'] for s in stats]
    pred_times = [s['predict_ms'] for s in stats]
    total_times = [s['total_ms'] for s in stats]

    print(f"    {'set_image (ms/img)':<25} mean={np.mean(set_times):>8.1f}  "
          f"std={np.std(set_times):>6.1f}  min={np.min(set_times):>7.1f}  max={np.max(set_times):>7.1f}")
    print(f"    {'predict  (ms/img)':<25} mean={np.mean(pred_times):>8.1f}  "
          f"std={np.std(pred_times):>6.1f}  min={np.min(pred_times):>7.1f}  max={np.max(pred_times):>7.1f}")
    print(f"    {'total    (ms/img)':<25} mean={np.mean(total_times):>8.1f}  "
          f"std={np.std(total_times):>6.1f}  min={np.min(total_times):>7.1f}  max={np.max(total_times):>7.1f}")
    print(f"    {'TOTAL (s)':<25} {result['total_time']:>8.3f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="SAM2 Pipeline 模拟 Benchmark: sequential vs batch"
    )
    parser.add_argument('--num-images', type=int, default=100,
                        help='测试图像数量 (default: 100)')
    parser.add_argument('--clusters-per-image', type=int, default=10,
                        help='每张图上的 cluster 数量 (default: 10)')
    parser.add_argument('--image-size', type=int, default=512,
                        help='图像尺寸 (default: 512)')
    parser.add_argument('--warmup', type=int, default=5,
                        help='GPU warmup 图像数量 (default: 5)')

    # SAM2 模型参数
    parser.add_argument('--sam-checkpoint', type=str,
                        default='./data/models/sam2/sam2.1_hiera_large.pt')
    parser.add_argument('--sam-config', type=str,
                        default='configs/sam2.1/sam2.1_hiera_l.yaml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output-dir', type=str, default='./output/benchmark_pipeline_sim')
    return parser.parse_args()


def main():
    args = parse_args()

    torch.autograd.set_grad_enabled(False)
    os.makedirs(args.output_dir, exist_ok=True)

    num_images = args.num_images
    clusters_per_image = args.clusters_per_image
    warmup_count = args.warmup
    total_images = num_images + warmup_count
    total_clusters = num_images * clusters_per_image

    print(f"\n{'='*75}")
    print(f"SAM2 Pipeline Benchmark: {num_images} images × {clusters_per_image} clusters")
    print(f"{'='*75}")
    print(f"  Images:           {num_images} (+ {warmup_count} warmup)")
    print(f"  Clusters/image:   {clusters_per_image}")
    print(f"  Total clusters:   {total_clusters}")
    print(f"  Image size:       {args.image_size}×{args.image_size}")
    print(f"  Device:           {args.device}")

    # ── 1. 加载模型 ──────────────────────────────────────────────
    print(f"\n[1/4] Loading SAM2 model...")
    t_load_start = time.perf_counter()
    predictor = load_sam2(args.sam_config, args.sam_checkpoint, args.device)
    t_load = time.perf_counter() - t_load_start
    print(f"  Model loaded in {t_load:.2f}s")

    # ── 2. 生成测试数据 ─────────────────────────────────────────
    print(f"\n[2/4] Generating {total_images} images × {clusters_per_image} prompts...")
    np.random.seed(42)
    images = generate_random_images(total_images, args.image_size)
    prompts_per_image = [generate_random_mask_prompts(clusters_per_image)
                         for _ in range(total_images)]

    warmup_images = images[:warmup_count]
    warmup_prompts = prompts_per_image[:warmup_count]
    test_images = images[warmup_count:]
    test_prompts = prompts_per_image[warmup_count:]

    # ── 3. Warmup ────────────────────────────────────────────────
    print(f"\n[3/4] GPU Warmup ({warmup_count} images)...")
    _ = run_pipeline_sequential(predictor, warmup_images, warmup_prompts)
    _ = run_pipeline_batch(predictor, warmup_images, warmup_prompts, batch_size=10)
    _ = run_pipeline_batch(predictor, warmup_images, warmup_prompts, batch_size=32)
    print(f"  Warmup done.")

    # ── 4. Benchmark ─────────────────────────────────────────────
    print(f"\n[4/4] Running benchmark ({num_images} images)...\n")

    # Sequential
    print(f"  --- Sequential (逐个 predict) ---")
    result_seq = run_pipeline_sequential(predictor, test_images, test_prompts)
    print_stats("Sequential", result_seq)

    # Batch size = 10
    print(f"\n  --- Batch (batch_size=10) ---")
    result_b10 = run_pipeline_batch(predictor, test_images, test_prompts, batch_size=10)
    print_stats("Batch(bs=10)", result_b10)

    # Batch size = 32
    print(f"\n  --- Batch (batch_size=32) ---")
    result_b32 = run_pipeline_batch(predictor, test_images, test_prompts, batch_size=32)
    print_stats("Batch(bs=32)", result_b32)

    # ── 汇总 ──────────────────────────────────────────────────────
    def avg(result, key):
        return np.mean([s[key] for s in result['per_image']])

    seq_total = result_seq['total_time']
    b10_total = result_b10['total_time']
    b32_total = result_b32['total_time']

    seq_per_img = avg(result_seq, 'total_ms')
    b10_per_img = avg(result_b10, 'total_ms')
    b32_per_img = avg(result_b32, 'total_ms')

    seq_set = avg(result_seq, 'set_image_ms')
    b10_set = avg(result_b10, 'set_image_ms')
    b32_set = avg(result_b32, 'set_image_ms')

    seq_pred = avg(result_seq, 'predict_ms')
    b10_pred = avg(result_b10, 'predict_ms')
    b32_pred = avg(result_b32, 'predict_ms')

    print(f"\n{'='*75}")
    print(f"BENCHMARK RESULTS  ({num_images} images × {clusters_per_image} clusters)")
    print(f"{'='*75}")

    print(f"\n  {'Metric':<30} {'Sequential':>12} {'Batch(10)':>12} {'Batch(32)':>12}")
    print(f"  {'-'*68}")
    print(f"  {'set_image (ms/img)':<30} {seq_set:>12.1f} {b10_set:>12.1f} {b32_set:>12.1f}")
    print(f"  {'predict  (ms/img)':<30} {seq_pred:>12.1f} {b10_pred:>12.1f} {b32_pred:>12.1f}")
    print(f"  {'total    (ms/img)':<30} {seq_per_img:>12.1f} {b10_per_img:>12.1f} {b32_per_img:>12.1f}")
    print(f"  {'TOTAL    (s)':<30} {seq_total:>12.3f} {b10_total:>12.3f} {b32_total:>12.3f}")

    print(f"\n  --- Speedup vs Sequential ---")
    print(f"  {'predict  加速比':<30} {'1.00x':>12} {seq_pred/b10_pred:>11.2f}x {seq_pred/b32_pred:>11.2f}x")
    print(f"  {'total    加速比':<30} {'1.00x':>12} {seq_per_img/b10_per_img:>11.2f}x {seq_per_img/b32_per_img:>11.2f}x")
    print(f"  {'总耗时   加速比':<30} {'1.00x':>12} {seq_total/b10_total:>11.2f}x {seq_total/b32_total:>11.2f}x")

    print(f"\n  --- 时间节省 ---")
    print(f"  {'vs Sequential 节省(s)':<30} {'—':>12} {seq_total - b10_total:>12.3f} {seq_total - b32_total:>12.3f}")
    print(f"  {'vs Sequential 节省(%)':<30} {'—':>12} {(1 - b10_total/seq_total)*100:>11.1f}% {(1 - b32_total/seq_total)*100:>11.1f}%")

    # 估算全切片
    print(f"\n  --- 全切片估算 (32,630 tiles × {clusters_per_image} clusters) ---")
    est_seq_h = seq_per_img * 32630 / 1000 / 3600
    est_b10_h = b10_per_img * 32630 / 1000 / 3600
    est_b32_h = b32_per_img * 32630 / 1000 / 3600
    print(f"  {'Sequential':<20} {est_seq_h:.2f} h")
    print(f"  {'Batch(bs=10)':<20} {est_b10_h:.2f} h")
    print(f"  {'Batch(bs=32)':<20} {est_b32_h:.2f} h")
    print(f"  {'节省(bs=10)':<20} {est_seq_h - est_b10_h:.2f} h ({(1-est_b10_h/est_seq_h)*100:.1f}%)")
    print(f"  {'节省(bs=32)':<20} {est_seq_h - est_b32_h:.2f} h ({(1-est_b32_h/est_seq_h)*100:.1f}%)")

    print(f"\n{'='*75}\n")

    # ── 保存结果 ──────────────────────────────────────────────────
    result_data = {
        'config': {
            'num_images': num_images,
            'clusters_per_image': clusters_per_image,
            'image_size': args.image_size,
            'warmup': warmup_count,
            'device': args.device,
        },
        'summary': {
            'sequential': {
                'total_s': seq_total,
                'mean_set_image_ms': seq_set,
                'mean_predict_ms': seq_pred,
                'mean_total_ms': seq_per_img,
            },
            'batch_10': {
                'total_s': b10_total,
                'mean_set_image_ms': b10_set,
                'mean_predict_ms': b10_pred,
                'mean_total_ms': b10_per_img,
                'speedup_predict': seq_pred / b10_pred,
                'speedup_total': seq_total / b10_total,
            },
            'batch_32': {
                'total_s': b32_total,
                'mean_set_image_ms': b32_set,
                'mean_predict_ms': b32_pred,
                'mean_total_ms': b32_per_img,
                'speedup_predict': seq_pred / b32_pred,
                'speedup_total': seq_total / b32_total,
            },
        },
        'per_image': {
            'sequential': result_seq['per_image'],
            'batch_10': result_b10['per_image'],
            'batch_32': result_b32['per_image'],
        },
    }

    json_path = os.path.join(args.output_dir, 'benchmark_result.json')
    with open(json_path, 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f"  Results saved to: {json_path}")

    # per-image CSV
    csv_path = os.path.join(args.output_dir, 'per_image_timing.csv')
    with open(csv_path, 'w') as f:
        f.write("image_idx,num_clusters,"
                "seq_set_image_ms,seq_predict_ms,seq_total_ms,"
                "b10_set_image_ms,b10_predict_ms,b10_total_ms,"
                "b32_set_image_ms,b32_predict_ms,b32_total_ms\n")
        for i in range(num_images):
            s = result_seq['per_image'][i]
            b10 = result_b10['per_image'][i]
            b32 = result_b32['per_image'][i]
            f.write(f"{i},{clusters_per_image},"
                    f"{s['set_image_ms']:.2f},{s['predict_ms']:.2f},{s['total_ms']:.2f},"
                    f"{b10['set_image_ms']:.2f},{b10['predict_ms']:.2f},{b10['total_ms']:.2f},"
                    f"{b32['set_image_ms']:.2f},{b32['predict_ms']:.2f},{b32['total_ms']:.2f}\n")
    print(f"  Per-image CSV: {csv_path}")


if __name__ == "__main__":
    main()
