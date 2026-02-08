import os
import sys
import argparse
from PIL import Image

def run_test(img_path=None):
    # Ensure the module can be found
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    try:
        from deepliif_inference import DeepLIIFInference
        print("Successfully imported DeepLIIFInference module.")
    except ImportError as e:
        print(f"Import Error: {e}")
        sys.exit(1)

    model_dir = './deepliif_models/'
    
    # 1. Initialize engine
    print(f"Initializing DeepLIIF engine with models from {model_dir}...")
    try:
        # Use GPU if available, else CPU
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        engine = DeepLIIFInference(model_dir=model_dir, device=device)
    except Exception as e:
        print(f"Initialization failed: {e}")
        return

    # 2. Check weight files
    weights = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
    print(f"Found {len(weights)} weight files in {model_dir}")
    
    # 3. Determine test image
    if not img_path:
        img_path = '/local1/yangwenjie/DeepLIIF/images/sample.png'
    
    if not os.path.exists(img_path):
        print(f"Test image not found at {img_path}. Creating a dummy image.")
        img = Image.new('RGB', (1024, 1024), color=(255, 255, 255))
        base_name = "dummy"
    else:
        print(f"Using test image: {img_path}")
        img = Image.open(img_path).convert('RGB')
        base_name = os.path.basename(img_path).split('.')[0]
    
    # 4. Run inference
    print(f"Running inference on {img_path}...")
    try:
        results = engine.inference(img)
        print("\nInference Results:")
        for name, result in results.items():
            if hasattr(result, 'size'):
                print(f" - {name}: {result.size} {result.mode}")
            else:
                print(f" - {name}: {type(result)}")
            
        # 5. Save results
        output_dir = './test_results/'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save specific modalities
        save_list = ['Seg', 'Hema', 'DAPI', 'Lap2', 'Marker', 'SegOverlaid', 'SegRefined']
        for mod in save_list:
            if mod in results:
                out_path = os.path.join(output_dir, f'{base_name}_{mod}.png')
                results[mod].save(out_path)
                print(f"Saved: {out_path}")
        
        print(f"\nTest completed successfully! Results in {output_dir}")
        
    except Exception as e:
        print(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, help='Path to test image')
    args = parser.parse_args()
    
    run_test(args.img)
