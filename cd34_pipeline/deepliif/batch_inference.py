#!/usr/bin/env python3
"""
batch_inference.py — DeepLIIF Batch Inference

Extends DeepLIIFInference with batch processing support.
Multiple 512x512 tiles are stacked and processed through
G1-G4, G51-G55 networks in a single forward pass.
"""

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from .inference import DeepLIIFInference
from .utils import (
    disable_batchnorm_tracking_stats,
    make_power_2,
    is_empty,
    get_empty_result_tiles,
)


def _get_batch_transform():
    """Transform for batch mode — does NOT add batch dim (unsqueeze)."""
    return transforms.Compose([
        transforms.Lambda(lambda i: make_power_2(i, base=4, method=Image.BICUBIC)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def _tensor_to_pil_single(tensor_3d: torch.Tensor) -> Image.Image:
    """Convert a single (3, H, W) tensor (range [-1, 1]) to PIL Image."""
    image_numpy = tensor_3d.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return Image.fromarray(image_numpy.astype(np.uint8))


def _batch_tensor_to_pil(batch_tensor: torch.Tensor) -> list:
    """Convert (N, C, H, W) batch tensor (range [-1, 1]) to list of PIL Images.

    Transfers the entire batch to CPU in ONE .cpu() call instead of one per image,
    reducing implicit GPU synchronization from N to 1.
    """
    batch_np = batch_tensor.cpu().float().numpy()  # single GPU→CPU transfer
    if batch_np.shape[1] == 1:
        batch_np = np.tile(batch_np, (1, 3, 1, 1))
    # (N, C, H, W) → (N, H, W, C), rescale [-1,1] → [0,255]
    batch_np = (np.transpose(batch_np, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
    batch_np = batch_np.clip(0, 255).astype(np.uint8)
    return [Image.fromarray(batch_np[i]) for i in range(batch_np.shape[0])]


class DeepLIIFBatchInference(DeepLIIFInference):
    """
    DeepLIIF with batch inference support.

    Processes multiple tiles in a single GPU forward pass for each generator
    network (G1-G4, G51-G55), significantly improving throughput.
    """

    def __init__(self, model_dir: str, device: str = 'cuda'):
        super().__init__(model_dir=model_dir, device=device)
        self._batch_transform = _get_batch_transform()

    @torch.no_grad()
    def inference_batch(
        self,
        images: list,
        batch_size: int = 4,
        seg_weights: list = None,
        resolution: str = '40x',
    ) -> list:
        """
        Batch inference on multiple PIL Images.

        Each batch of ``batch_size`` tiles is stacked and sent through
        the networks in a single forward pass.  Empty / background tiles
        receive placeholder results without consuming GPU time.

        Args:
            images: List of PIL Images (each should be tile_size x tile_size).
            batch_size: Max number of tiles per GPU forward pass.
            seg_weights: Segmentation aggregation weights [G51..G55].
            resolution: Microscope resolution (unused here, kept for API compat).

        Returns:
            List[dict]:  One result dict per input image.
                         Keys: 'Hema', 'DAPI', 'Lap2', 'Marker', 'Seg'.
        """
        total = len(images)
        all_results: list = [None] * total

        # 1. Identify empty (background) tiles
        non_empty_indices: list[int] = []
        for i, img in enumerate(images):
            if is_empty(img):
                all_results[i] = self._make_empty_result()
            else:
                non_empty_indices.append(i)

        if not non_empty_indices:
            return all_results

        # 2. Process non-empty tiles in batches
        for batch_start in range(0, len(non_empty_indices), batch_size):
            batch_indices = non_empty_indices[batch_start:batch_start + batch_size]
            batch_results = self._forward_batch(
                [images[i] for i in batch_indices],
                seg_weights=seg_weights,
            )
            for j, orig_idx in enumerate(batch_indices):
                all_results[orig_idx] = batch_results[j]

        return all_results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _forward_batch(self, imgs: list, seg_weights: list = None) -> list:
        """Run a batch of non-empty images through G-networks."""
        # Stack tensors: each transform -> (3, H, W), then stack -> (N, 3, H, W)
        tensors = [self._batch_transform(img) for img in imgs]
        batch_ts = torch.stack(tensors, dim=0).to(self.device)  # (N, 3, H, W)

        # G1-G4 modality translation
        mod_outputs: dict[str, torch.Tensor] = {}
        for name in self.modality_names:
            if name in self.nets:
                mod_outputs[name] = self.nets[name](batch_ts)  # (N, 3, H, W)

        # G51-G55 segmentation
        seg_outputs: list[torch.Tensor] = []
        if 'G51' in self.nets:
            seg_outputs.append(self.nets['G51'](batch_ts))
        for i, mod_name in enumerate(self.modality_names):
            seg_name = f'G5{i + 2}'
            if seg_name in self.nets and mod_name in mod_outputs:
                seg_outputs.append(self.nets[seg_name](mod_outputs[mod_name]))

        # Aggregate segmentation
        if seg_weights is None:
            weights = [0.2, 0.2, 0.2, 0.2, 0.2]
        else:
            weights = seg_weights

        final_seg: torch.Tensor | None = None
        if seg_outputs:
            final_seg = torch.zeros_like(seg_outputs[0])
            for s, w in zip(seg_outputs, weights):
                final_seg += s * w

        # Batch GPU→CPU transfer: 5 .cpu() calls instead of N*5
        n = batch_ts.shape[0]
        semantic_map = [('Hema', 'G1'), ('DAPI', 'G2'), ('Lap2', 'G3'), ('Marker', 'G4')]

        mod_pils: dict[str, list] = {}
        for g_key, tensor in mod_outputs.items():
            mod_pils[g_key] = _batch_tensor_to_pil(tensor)
        seg_pils = _batch_tensor_to_pil(final_seg) if final_seg is not None else None

        results_list: list[dict] = []
        for idx in range(n):
            result: dict = {}
            for semantic_key, g_key in semantic_map:
                if g_key in mod_pils:
                    result[semantic_key] = mod_pils[g_key][idx]
            if seg_pils is not None:
                result['Seg'] = seg_pils[idx]
            results_list.append(result)

        return results_list

    @staticmethod
    def _make_empty_result() -> dict:
        """Return placeholder result dict for empty / background tiles."""
        empty = get_empty_result_tiles(512)
        return {
            'Hema': empty.get('Hema'),
            'DAPI': empty.get('DAPI'),
            'Lap2': empty.get('Lap2'),
            'Marker': empty.get('Marker'),
            'Seg': empty.get('Seg'),
        }
