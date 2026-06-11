import os
import argparse
import time
import sys
import csv
import numpy as np
import ncnn
from PIL import Image

# For tracking memory on Linux / Raspberry Pi
try:
    import resource
except ImportError:
    resource = None

# Configuration for known models
MODEL_CONFIGS = {
    'nanodet_model_best': {
        'size': (320, 320),
        'outputs': ['out0']
    },
    'effdet_rezipped': {
        'size': (512, 512),
        'outputs': [f'out{i}' for i in range(10)]
    },
    'yolov8_best': {
        'size': (512, 512),
        'outputs': ['out0']
    },
    'yolov9_best': {
        'size': (512, 512),
        'outputs': ['out0']
    },
    'yolov10_best': {
        'size': (512, 512),
        'outputs': ['out0']
    },
    'yolov11_best': {
        'size': (512, 512),
        'outputs': ['out0']
    }
}

def get_peak_memory():
    """Returns peak memory usage of the process in MB."""
    if resource is not None:
        # On Linux/macOS, ru_maxrss is in kilobytes (KB)
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return 0.0

def preprocess_image(image_path, target_size):
    """Loads and preprocesses an image for inference."""
    try:
        img = Image.open(image_path).convert('RGB')
        # Resize to target size (width, height)
        img = img.resize(target_size)
        # Convert to numpy array and normalize if needed (here we follow the 0-1 range float format)
        img_np = np.array(img, dtype=np.float32) / 255.0
        # HWC to CHW (ncnn expects CHW)
        img_np = img_np.transpose((2, 0, 1))
        # Add batch dimension
        return np.expand_dims(img_np, axis=0)
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None

def run_inference(model_name, param_path, bin_path, input_tensor, outputs_list, benchmark=False, num_runs=100):
    """Loads the NCNN model and runs inference with the input tensor."""
    print(f"\n--- Running inference on: {model_name} ---")
    print(f"Param path: {param_path}")
    print(f"Bin path: {bin_path}")
    print(f"Input shape: {list(input_tensor.shape)}")

    # Calculate model file sizes
    param_size = os.path.getsize(param_path) / (1024 * 1024)
    bin_size = os.path.getsize(bin_path) / (1024 * 1024)
    total_size = param_size + bin_size
    print(f"Model File Size: {total_size:.2f} MB (Param: {param_size:.2f} MB, Bin: {bin_size:.2f} MB)")

    out_tensors = []
    metrics = {
        'model': model_name,
        'size_mb': round(total_size, 3),
        'avg_latency_ms': 0.0,
        'std_latency_ms': 0.0,
        'fps': 0.0,
        'ram_mb': 0.0,
        'status': 'success'
    }

    try:
        with ncnn.Net() as net:
            net.load_param(param_path)
            net.load_model(bin_path)

            # Warmup runs
            with net.create_extractor() as ex:
                in_mat = ncnn.Mat(input_tensor[0]).clone()
                ex.input("in0", in_mat)
                for out_name in outputs_list:
                    ex.extract(out_name)

            if benchmark:
                print(f"Benchmarking efficiency over {num_runs} runs...")
                latencies = []
                mem_start = get_peak_memory()

                for _ in range(num_runs):
                    start_time = time.perf_counter()
                    with net.create_extractor() as ex:
                        in_mat = ncnn.Mat(input_tensor[0]).clone()
                        ex.input("in0", in_mat)
                        for out_name in outputs_list:
                            ex.extract(out_name)
                    latencies.append((time.perf_counter() - start_time) * 1000.0) # in ms

                mem_end = get_peak_memory()
                avg_latency = np.mean(latencies)
                std_latency = np.std(latencies)
                fps = 1000.0 / avg_latency
                ram_used = max(0.0, mem_end - mem_start)

                metrics['avg_latency_ms'] = round(avg_latency, 3)
                metrics['std_latency_ms'] = round(std_latency, 3)
                metrics['fps'] = round(fps, 2)
                metrics['ram_mb'] = round(mem_end, 2)

                print(f"Benchmark Results for {model_name}:")
                print(f"  Avg Latency: {avg_latency:.2f} ms (Std Dev: {std_latency:.2f} ms)")
                print(f"  Throughput:  {fps:.2f} FPS")
                if resource is not None:
                    print(f"  Peak RAM usage: {mem_end:.2f} MB (Incremental: {ram_used:.2f} MB)")
                else:
                    print(f"  RAM tracking: Not supported on this OS (requires resource module on Linux/Pi)")

            else:
                # Standard single run execution
                with net.create_extractor() as ex:
                    in_mat = ncnn.Mat(input_tensor[0]).clone()
                    ex.input("in0", in_mat)

                    for out_name in outputs_list:
                        ret, out_mat = ex.extract(out_name)
                        if ret == 0:
                            out_np = np.array(out_mat)
                            out_tensor = np.expand_dims(out_np, axis=0)
                            out_tensors.append((out_name, out_tensor))
                        else:
                            print(f"Warning: Failed to extract output '{out_name}' (error code: {ret})")
            
                # Display output summary
                for name, tensor in out_tensors:
                    print(f"Output '{name}' shape: {list(tensor.shape)}")
                    print(f"  Min: {tensor.min():.4f}, Max: {tensor.max():.4f}, Mean: {tensor.mean():.4f}")
            
        return metrics

    except Exception as e:
        print(f"Error running inference for {model_name}: {e}")
        metrics['status'] = f"failed: {str(e)}"
        return metrics

def main():
    parser = argparse.ArgumentParser(description="Run/Benchmark inference on NCNN models in best_weights.")
    parser.add_argument('--weights_dir', type=str, default=None, 
                        help="Path to weights directory containing NCNN files. Auto-detects if not provided.")
    parser.add_argument('--image', type=str, default=None, 
                        help="Path to an input image or a directory of images to run inference on. If not provided, runs with dummy inputs.")
    parser.add_argument('--benchmark', action='store_true',
                        help="Runs the model multiple times to benchmark latency, FPS, and resource utilization.")
    parser.add_argument('--num_runs', type=int, default=100,
                        help="Number of iterations to run for benchmarking.")
    parser.add_argument('--csv', type=str, default=None,
                        help="Output path for CSV file to store benchmark results.")
    
    args = parser.parse_args()

    # Locate weights directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    potential_dirs = [
        args.weights_dir,
        os.path.join(script_dir, '..', '..', 'best_weights'),
        os.path.abspath(os.path.join(script_dir, '../../../best_weights')),
        'best_weights'
    ]
    
    weights_dir = None
    for d in potential_dirs:
        if d and os.path.exists(d):
            weights_dir = os.path.abspath(d)
            break

    if not weights_dir:
        print("Error: Could not locate the 'best_weights' directory. Please specify it using --weights_dir.")
        sys.exit(1)

    print(f"Using weights directory: {weights_dir}")

    # Find all NCNN model param files
    param_files = [f for f in os.listdir(weights_dir) if f.endswith('.ncnn.param')]
    
    if not param_files:
        print("No NCNN (.ncnn.param) models found in the weights directory.")
        sys.exit(0)

    print(f"Found {len(param_files)} NCNN model(s) to process.")

    # Find all images
    image_paths = []
    if args.image:
        if os.path.isdir(args.image):
            valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
            image_paths = [os.path.join(args.image, f) for f in os.listdir(args.image) if f.lower().endswith(valid_exts)]
        else:
            image_paths = [args.image]
    else:
        image_paths = [None]

    print(f"Running on {len(image_paths)} image(s) source.")

    results = []

    for pf in param_files:
        base_name = pf.replace('.ncnn.param', '')
        param_path = os.path.join(weights_dir, pf)
        bin_path = os.path.join(weights_dir, base_name + '.ncnn.bin')

        if not os.path.exists(bin_path):
            print(f"Warning: Bin file not found for {pf}, skipping.")
            continue

        # Get config
        config = MODEL_CONFIGS.get(base_name, {'size': (512, 512), 'outputs': ['out0']})
        target_size = config['size']
        outputs_list = config['outputs']

        # Process each image for this model
        for img_path in image_paths:
            if img_path:
                input_tensor = preprocess_image(img_path, target_size)
                if input_tensor is None:
                    print(f"Skipping {base_name} on {img_path} due to image loading error.")
                    continue
                img_name = os.path.basename(img_path)
            else:
                # Generate dummy input (random noise) using numpy
                np.random.seed(0)
                input_tensor = np.random.rand(1, 3, target_size[1], target_size[0]).astype(np.float32)
                img_name = "dummy_input"

            res = run_inference(base_name, param_path, bin_path, input_tensor, outputs_list, benchmark=args.benchmark, num_runs=args.num_runs)
            if res:
                res['image'] = img_name
                results.append(res)

    if args.csv and results:
        # Write results to CSV
        keys = ['model', 'image', 'size_mb', 'avg_latency_ms', 'std_latency_ms', 'fps', 'ram_mb', 'status']
        with open(args.csv, 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, keys)
            dict_writer.writeheader()
            # Filter and order the dicts based on keys
            filtered_results = [{k: row.get(k, '') for k in keys} for row in results]
            dict_writer.writerows(filtered_results)
        print(f"\nSuccessfully saved all benchmarking results to: {args.csv}")

if __name__ == '__main__':
    main()
