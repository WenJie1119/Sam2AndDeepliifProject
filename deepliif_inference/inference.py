import os
import torch
from PIL import Image
from .utils import disable_batchnorm_tracking_stats, get_transform, tensor_to_pil, InferenceTiler

class DeepLIIFInference:
    """
    Standalone DeepLIIF Inference class.
    Handles loading models and performing inference on images.
    """
    
    def __init__(self, model_dir, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device)
        self.model_dir = model_dir
        self.transform = get_transform()
        self.nets = {}
        
        # Core generators
        self.modality_names = ['G1', 'G2', 'G3', 'G4']
        # Segmentation generators
        self.segmentation_names = ['G51', 'G52', 'G53', 'G54', 'G55']
        
        self._load_models()
        
    def _load_models(self):
        """Load TorchScript models from model_dir."""
        all_names = self.modality_names + self.segmentation_names
        for name in all_names:
            model_path = os.path.join(self.model_dir, f'{name}.pt')
            if not os.path.exists(model_path):
                print(f"Warning: Model file {model_path} not found. Skipping {name}.")
                continue
            
            # Load serialized model
            net = torch.jit.load(model_path, map_location=self.device)
            net = disable_batchnorm_tracking_stats(net)
            net.eval()
            self.nets[name] = net
            print(f"Loaded {name} model.")

    @torch.no_grad()
    def inference(self, img, seg_weights=None, resolution='40x', do_postprocessing=True, tile_size=512,
                  seg_thresh=120, size_thresh='default', marker_thresh=None, 
                  size_thresh_upper=None, noise_thresh=4, large_noise_thresh=None,
                  color_dapi=False, color_marker=False):
        """
        Perform DeepLIIF inference on a single PIL Image.
        Supports automatic tiling for large images.
        Returns a dictionary of result PIL Images.
        
        Args:
            img: PIL Image to process
            seg_weights: List of 5 weights for segmentation aggregation (G51-G55)
            resolution: Microscope resolution ('10x', '20x', '40x')
            do_postprocessing: Whether to run post-processing
            tile_size: Tile size for processing large images
            seg_thresh: Segmentation threshold (default 120)
            size_thresh: Cell size threshold ('default' or int)
            marker_thresh: Marker threshold for positive/negative classification
            size_thresh_upper: Upper bound for cell size filtering
            noise_thresh: Noise threshold for small debris (default 4)
            large_noise_thresh: Large noise threshold to filter out large objects ('default', None, or int)
            color_dapi: Apply cyan/blue pseudo-coloring to DAPI output (default False)
            color_marker: Apply yellow/brown pseudo-coloring to Marker output (default False)
        """
        # Initialize Tiler
        # Overlap size typically tile_size // 16 in DeepLIIF
        overlap_size = tile_size // 16
        tiler = InferenceTiler(img, tile_size, overlap_size)
        
        # Iterate over tiles
        for tile in tiler:
            # 1. Preprocess tile
            ts = self.transform(tile).to(self.device)
            
            # 2. Modality Translation (G1-G4)
            mod_results = {}
            tile_results = {}
            for name in self.modality_names:
                if name in self.nets:
                    mod_results[name] = self.nets[name](ts)
                    tile_results[name] = tensor_to_pil(mod_results[name])
            
            # 3. Individual Segmentations (G51-G55)
            seg_results_ts = []
            
            if 'G51' in self.nets:
                seg_results_ts.append(self.nets['G51'](ts))
                
            for i, mod_name in enumerate(self.modality_names):
                seg_name = f'G5{i+2}'
                if seg_name in self.nets and mod_name in mod_results:
                    seg_results_ts.append(self.nets[seg_name](mod_results[mod_name]))
            
            # 4. Aggregate Segmentation
            if seg_results_ts:
                if seg_weights is None:
                    weight = 1.0 / len(seg_results_ts)
                    weights = [weight] * len(seg_results_ts)
                else:
                    weights = seg_weights
                
                final_seg = torch.zeros_like(seg_results_ts[0])
                for s, w in zip(seg_results_ts, weights):
                    final_seg += s * w
                
                tile_results['Seg'] = tensor_to_pil(final_seg)
            
            # Stitch this tile's results
            tiler.stitch(tile_results)
            
        # Get final stitched results
        results = tiler.results()
        
        # Add semantic aliases after stitching
        semantic_results = {
            'Hema': results.get('G1'),
            'DAPI': results.get('G2'),
            'Lap2': results.get('G3'),
            'Marker': results.get('G4'),
            'Seg': results.get('Seg')
        }
        semantic_results = {k: v for k, v in semantic_results.items() if v is not None}
        results = semantic_results  # Only keep semantic names, not G1-G4 duplicates
        
        # Apply colorization (optional)
        if color_dapi and 'DAPI' in results:
            # Cyan/Blue coloring: R=0, G=gray, B=gray
            matrix = (       0,        0,        0, 0,
                      299/1000, 587/1000, 114/1000, 0,
                      299/1000, 587/1000, 114/1000, 0)
            results['DAPI'] = results['DAPI'].convert('RGB', matrix)
        
        if color_marker and 'Marker' in results:
            # Yellow/Brown coloring: R=gray, G=gray, B=0
            matrix = (299/1000, 587/1000, 114/1000, 0,
                      299/1000, 587/1000, 114/1000, 0,
                             0,        0,        0, 0)
            results['Marker'] = results['Marker'].convert('RGB', matrix)
        
        # 5. Post-processing (Optional but recommended)
        if do_postprocessing and results.get('Seg'):
            from .postprocessing import compute_final_results
            
            # Prepare inputs for post-processing
            # Need original image (as array), segmentation (as array), marker (as array)
            # Markers is G4
            marker_img = results.get('Marker')
            seg_img = results.get('Seg')
            
            if marker_img and seg_img:
                try:
                    overlay, refined, scoring = compute_final_results(
                        orig=img,
                        seg=seg_img,
                        marker=marker_img,
                        resolution=resolution,
                        seg_thresh=seg_thresh,
                        size_thresh=size_thresh,
                        marker_thresh=marker_thresh,
                        size_thresh_upper=size_thresh_upper,
                        noise_thresh=noise_thresh,
                        large_noise_thresh=large_noise_thresh
                    )
                    results['SegOverlaid'] = Image.fromarray(overlay)
                    results['SegRefined'] = Image.fromarray(refined)
                    results['Scoring'] = scoring
                except Exception as e:
                    print(f"Warning: Post-processing failed: {e}")
            
        return results

def inference_single_image(img_path, model_dir, device='cuda', resolution='40x'):
    """Helper function for quick inference."""
    img = Image.open(img_path).convert('RGB')
    engine = DeepLIIFInference(model_dir, device)
    return engine.inference(img, resolution=resolution)
