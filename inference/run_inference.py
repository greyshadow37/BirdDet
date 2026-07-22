import os
import argparse
import time
import sys
import csv
import glob
import numpy as np
from PIL import Image
import psutil

# Configuration for known models (fallback sizes)
MODEL_CONFIGS = {
    'nanodet': {
        'size': (416, 416),  # Standard NanoDet models use 416x416 or 320x320
        'outputs': ['out0']
    },
    'rt-detr': {
        'size': (640, 640),  # RT-DETR defaults
        'outputs': ['out0']
    },
    'yolo': {
        'size': (512, 512),
        'outputs': ['out0']
    }
}

def get_peak_memory():
    """Returns peak memory usage (RSS) of the process in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024.0 * 1024.0)

def get_model_fallback_info(model_path):
    """Determines fallback config details based on filename."""
    basename = os.path.basename(model_path).lower()
    if 'nanodet' in basename:
        return MODEL_CONFIGS['nanodet']['size'], MODEL_CONFIGS['nanodet']['outputs']
    elif 'rt-detr' in basename:
        return MODEL_CONFIGS['rt-detr']['size'], MODEL_CONFIGS['rt-detr']['outputs']
    else:
        return MODEL_CONFIGS['yolo']['size'], MODEL_CONFIGS['yolo']['outputs']

def preprocess_image(image_path, target_size):
    """Loads and preprocesses an image for inference."""
    try:
        if not image_path:
            return None
        if os.path.isdir(image_path):
            print(f"Warning: '{image_path}' is a directory, not an image file. Skipping.")
            return None
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size)
        img_np = np.array(img, dtype=np.float32) / 255.0
        # HWC to CHW (ncnn/onnx expect CHW)
        img_np = img_np.transpose((2, 0, 1))
        return np.expand_dims(img_np, axis=0)
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None

def run_ncnn_inference(model_name, param_path, bin_path, image_path, fallback_size, outputs_list, benchmark=False, num_runs=100):
    """Runs inference using NCNN backend."""
    metrics = {
        'model': model_name,
        'format': 'NCNN',
        'size_mb': 0.0,
        'avg_latency_ms': 0.0,
        'std_latency_ms': 0.0,
        'fps': 0.0,
        'ram_mb': 0.0,
        'status': 'success'
    }
    
    try:
        import ncnn
    except ImportError:
        metrics['status'] = 'failed: ncnn library is not installed'
        return metrics

    try:
        total_size = (os.path.getsize(param_path) + os.path.getsize(bin_path)) / (1024 * 1024)
        metrics['size_mb'] = round(total_size, 3)
        
        # Preprocess input image using fallback size
        if image_path:
            input_tensor = preprocess_image(image_path, fallback_size)
        else:
            input_tensor = np.random.rand(1, 3, fallback_size[1], fallback_size[0]).astype(np.float32)
            
        if input_tensor is None:
            raise ValueError("Input tensor preprocessing failed.")

        with ncnn.Net() as net:
            net.load_param(param_path)
            net.load_model(bin_path)
            
            # Warmup run
            with net.create_extractor() as ex:
                in_mat = ncnn.Mat(input_tensor[0]).clone()
                ex.input("in0", in_mat)
                for out_name in outputs_list:
                    ex.extract(out_name)
            
            if benchmark:
                latencies = []
                mem_start = get_peak_memory()
                for _ in range(num_runs):
                    t0 = time.perf_counter()
                    with net.create_extractor() as ex:
                        in_mat = ncnn.Mat(input_tensor[0]).clone()
                        ex.input("in0", in_mat)
                        for out_name in outputs_list:
                            ex.extract(out_name)
                    latencies.append((time.perf_counter() - t0) * 1000.0)
                mem_end = get_peak_memory()
                
                avg_lat = np.mean(latencies)
                metrics['avg_latency_ms'] = round(avg_lat, 3)
                metrics['std_latency_ms'] = round(np.std(latencies), 3)
                metrics['fps'] = round(1000.0 / avg_lat, 2)
                metrics['ram_mb'] = round(mem_end, 2)
        return metrics
    except Exception as e:
        metrics['status'] = f"failed: {str(e)}"
        return metrics

def run_onnx_inference(model_name, model_path, image_path, fallback_size, benchmark=False, num_runs=100):
    """Runs inference using ONNX Runtime backend with dynamic input shape detection."""
    metrics = {
        'model': model_name,
        'format': 'ONNX',
        'size_mb': 0.0,
        'avg_latency_ms': 0.0,
        'std_latency_ms': 0.0,
        'fps': 0.0,
        'ram_mb': 0.0,
        'status': 'success'
    }
    
    try:
        import onnxruntime as ort
    except ImportError:
        metrics['status'] = 'failed: onnxruntime library is not installed'
        return metrics

    try:
        total_size = os.path.getsize(model_path) / (1024 * 1024)
        metrics['size_mb'] = round(total_size, 3)
        
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        input_meta = session.get_inputs()[0]
        input_name = input_meta.name
        input_type = input_meta.type
        input_shape = input_meta.shape
        output_names = [o.name for o in session.get_outputs()]
        
        # Detect target input size dynamically
        target_size = fallback_size
        if len(input_shape) == 4:
            h, w = input_shape[2], input_shape[3]
            if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
                target_size = (w, h)
                
        # Preprocess input image using detected size
        if image_path:
            input_tensor = preprocess_image(image_path, target_size)
        else:
            input_tensor = np.random.rand(1, 3, target_size[1], target_size[0]).astype(np.float32)
            
        if input_tensor is None:
            raise ValueError("Input tensor preprocessing failed.")
            
        # Adjust input dtype dynamically
        if 'uint8' in input_type:
            inp = (input_tensor * 255.0).astype(np.uint8)
        elif 'int8' in input_type:
            inp = ((input_tensor * 255.0) - 128).astype(np.int8)
        else:
            inp = input_tensor.astype(np.float32)
            
        # Warmup run
        session.run(output_names, {input_name: inp})
        
        if benchmark:
            latencies = []
            mem_start = get_peak_memory()
            for _ in range(num_runs):
                t0 = time.perf_counter()
                session.run(output_names, {input_name: inp})
                latencies.append((time.perf_counter() - t0) * 1000.0)
            mem_end = get_peak_memory()
            
            avg_lat = np.mean(latencies)
            metrics['avg_latency_ms'] = round(avg_lat, 3)
            metrics['std_latency_ms'] = round(np.std(latencies), 3)
            metrics['fps'] = round(1000.0 / avg_lat, 2)
            metrics['ram_mb'] = round(mem_end, 2)
        return metrics
    except Exception as e:
        metrics['status'] = f"failed: {str(e)}"
        return metrics

def run_tflite_inference(model_name, model_path, image_path, fallback_size, benchmark=False, num_runs=100):
    """Runs inference using TensorFlow Lite backend with dynamic input shape detection."""
    metrics = {
        'model': model_name,
        'format': 'TFLite',
        'size_mb': 0.0,
        'avg_latency_ms': 0.0,
        'std_latency_ms': 0.0,
        'fps': 0.0,
        'ram_mb': 0.0,
        'status': 'success'
    }

    try:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    except ImportError:
        try:
            import tflite_runtime.interpreter as tflite
            Interpreter = tflite.Interpreter
        except ImportError:
            metrics['status'] = "failed: neither 'tensorflow' nor 'tflite-runtime' is installed"
            return metrics

    try:
        total_size = os.path.getsize(model_path) / (1024 * 1024)
        metrics['size_mb'] = round(total_size, 3)

        # Disable XNNPACK delegate if running NanoDet INT8 to prevent Windows crash
        if 'nanodet' in model_name.lower() and 'int8' in model_name.lower():
            interpreter = Interpreter(model_path=model_path, num_threads=4)
        else:
            interpreter = Interpreter(model_path=model_path)
            
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        input_shape = input_details[0]['shape']
        input_dtype = input_details[0]['dtype']
        
        # Detect target input size dynamically
        target_size = fallback_size
        if len(input_shape) == 4:
            if input_shape[3] == 3:
                h, w = input_shape[1], input_shape[2]
            else:
                h, w = input_shape[2], input_shape[3]
            if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
                target_size = (w, h)
                
        # Preprocess input image using detected size
        if image_path:
            input_tensor = preprocess_image(image_path, target_size)
        else:
            input_tensor = np.random.rand(1, 3, target_size[1], target_size[0]).astype(np.float32)
            
        if input_tensor is None:
            raise ValueError("Input tensor preprocessing failed.")
            
        # Check BHWC vs BCHW
        if len(input_shape) == 4 and input_shape[3] == 3:
            tflite_input = np.transpose(input_tensor, (0, 2, 3, 1))
        else:
            tflite_input = input_tensor
            
        # Cast dtype dynamically
        if input_dtype == np.uint8 or (hasinstance := isinstance(input_dtype, type) and input_dtype.__name__ == 'uint8'):
            tflite_input = (tflite_input * 255.0).astype(np.uint8)
        elif input_dtype == np.int8 or (hasinstance := isinstance(input_dtype, type) and input_dtype.__name__ == 'int8'):
            tflite_input = ((tflite_input * 255.0) - 128).astype(np.int8)
        else:
            tflite_input = tflite_input.astype(np.float32)
            
        # Warmup run
        interpreter.set_tensor(input_details[0]['index'], tflite_input)
        interpreter.invoke()
        
        if benchmark:
            latencies = []
            mem_start = get_peak_memory()
            for _ in range(num_runs):
                t0 = time.perf_counter()
                interpreter.set_tensor(input_details[0]['index'], tflite_input)
                interpreter.invoke()
                latencies.append((time.perf_counter() - t0) * 1000.0)
            mem_end = get_peak_memory()
            
            avg_lat = np.mean(latencies)
            metrics['avg_latency_ms'] = round(avg_lat, 3)
            metrics['std_latency_ms'] = round(np.std(latencies), 3)
            metrics['fps'] = round(1000.0 / avg_lat, 2)
            metrics['ram_mb'] = round(mem_end, 2)
        return metrics
    except Exception as e:
        metrics['status'] = f"failed: {str(e)}"
        return metrics

def main():
    parser = argparse.ArgumentParser(description="Unified multi-backend inference benchmarking.")
    parser.add_argument('--weights_dir', type=str, default=None, 
                        help="Path to the 'best_weights' directory.")
    parser.add_argument('--image', type=str, default=None, 
                        help="Path to an input image. If not provided, dummy input is used.")
    parser.add_argument('--benchmark', action='store_true',
                        help="Runs the model multiple times to benchmark latency, FPS, and RAM.")
    parser.add_argument('--num_runs', type=int, default=100,
                        help="Number of runs for benchmarking.")
    parser.add_argument('--csv', type=str, default=None,
                        help="Output path for CSV file to store benchmark results.")
    parser.add_argument('--resume', action='store_true',
                        help="Skip models that already ran successfully in the output CSV.")
    
    args = parser.parse_args()
    
    # Load existing CSV results if resuming
    existing_results = {}
    if args.resume and args.csv and os.path.exists(args.csv):
        try:
            with open(args.csv, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = (row['model'], row['format'], row['image'])
                    existing_results[key] = {
                        'model': row['model'],
                        'format': row['format'],
                        'image': row['image'],
                        'size_mb': float(row['size_mb']) if row['size_mb'] else 0.0,
                        'avg_latency_ms': float(row['avg_latency_ms']) if row['avg_latency_ms'] else 0.0,
                        'std_latency_ms': float(row['std_latency_ms']) if row['std_latency_ms'] else 0.0,
                        'fps': float(row['fps']) if row['fps'] else 0.0,
                        'ram_mb': float(row['ram_mb']) if row['ram_mb'] else 0.0,
                        'status': row['status']
                    }
            print(f"Loaded {len(existing_results)} existing results to resume from.")
        except Exception as e:
            print(f"Warning: Could not read existing CSV to resume: {e}")

def discover_models_in_directory(weights_dir):
    """
    Discovers ONNX, TFLite, and NCNN model files inside weights_dir.
    Supports standard layout (weights_dir/onnx, weights_dir/tflite, etc.)
    and recursive nested architecture layout (weights_dir/<arch>/<precision>/...).
    """
    onnx_models = []
    tflite_models = []
    ncnn_models = []

    if not weights_dir or not os.path.exists(weights_dir):
        return onnx_models, tflite_models, ncnn_models

    # --- 1. ONNX ---
    # Standard subdirs first
    for sub in ['onnx', 'onnx_int8']:
        o_dir = os.path.join(weights_dir, sub)
        if os.path.exists(o_dir):
            for of in glob.glob(os.path.join(o_dir, '*.onnx')):
                onnx_models.append((sub, of))

    # Recursive scan for additional ONNX files
    added_onnx_paths = {p for _, p in onnx_models}
    for root, _, files in os.walk(weights_dir):
        for f in files:
            if f.endswith('.onnx') and not f.endswith('.pnnx.onnx'):
                full_p = os.path.abspath(os.path.join(root, f))
                if full_p not in added_onnx_paths:
                    rel_sub = 'onnx_int8' if 'int8' in full_p.lower() else 'onnx'
                    onnx_models.append((rel_sub, full_p))
                    added_onnx_paths.add(full_p)

    # --- 2. TFLite ---
    for sub in ['tflite', 'tflite_int8']:
        tf_dir = os.path.join(weights_dir, sub)
        if os.path.exists(tf_dir):
            for tf_file in glob.glob(os.path.join(tf_dir, '**', '*.tflite'), recursive=True):
                tflite_models.append((sub, tf_file))
            for f in os.listdir(tf_dir):
                full_p = os.path.join(tf_dir, f)
                if os.path.isfile(full_p) and 'tflite' in f.lower() and not f.endswith('.tflite'):
                    tflite_models.append((sub, full_p))

    added_tflite_paths = {p for _, p in tflite_models}
    ignore_exts = ('.json', '.pb', '.fbs', '.py', '.txt', '.sh', '.index', '.yaml', '.data-00000-of-00001')
    for root, _, files in os.walk(weights_dir):
        for f in files:
            full_p = os.path.abspath(os.path.join(root, f))
            if (f.endswith('.tflite') or ('tflite' in f.lower() and os.path.isfile(full_p))) and not f.endswith(ignore_exts):
                if full_p not in added_tflite_paths and os.path.isfile(full_p):
                    rel_sub = 'tflite_int8' if 'int8' in full_p.lower() else 'tflite'
                    tflite_models.append((rel_sub, full_p))
                    added_tflite_paths.add(full_p)

    # --- 3. NCNN ---
    ncnn_dir = os.path.join(weights_dir, 'ncnn')
    if os.path.exists(ncnn_dir):
        subdirs = [os.path.join(ncnn_dir, d) for d in os.listdir(ncnn_dir) if os.path.isdir(os.path.join(ncnn_dir, d))]
        for sd in subdirs:
            for pf in glob.glob(os.path.join(sd, '*.param')):
                bin_file = pf.replace('.param', '.bin')
                if os.path.exists(bin_file):
                    ncnn_models.append((os.path.basename(sd), pf, bin_file))

    added_ncnn_params = {pf for _, pf, _ in ncnn_models}
    for root, _, files in os.walk(weights_dir):
        for f in files:
            if f.endswith('.param') and not f.endswith('.pnnx.param'):
                pf = os.path.abspath(os.path.join(root, f))
                if pf not in added_ncnn_params:
                    bin_file = pf.replace('.param', '.bin')
                    if os.path.exists(bin_file):
                        base_name = os.path.basename(root)
                        if base_name in ('fp32', 'int8'):
                            parent_name = os.path.basename(os.path.dirname(root))
                            model_name = f"{parent_name}_{base_name}"
                        else:
                            model_name = base_name
                        ncnn_models.append((model_name, pf, bin_file))
                        added_ncnn_params.add(pf)


    return onnx_models, tflite_models, ncnn_models

def main():
    parser = argparse.ArgumentParser(description="Unified multi-backend inference benchmarking.")
    parser.add_argument('--weights_dir', type=str, default=None, 
                        help="Path to the weights directory (e.g. 'best_weights' or 'results/architecture/best_weights').")
    parser.add_argument('--image', type=str, default=None, 
                        help="Path to an input image. If not provided, dummy input is used.")
    parser.add_argument('--benchmark', action='store_true',
                        help="Runs the model multiple times to benchmark latency, FPS, and RAM.")
    parser.add_argument('--num_runs', type=int, default=100,
                        help="Number of runs for benchmarking.")
    parser.add_argument('--csv', type=str, default=None,
                        help="Output path for CSV file to store benchmark results.")
    parser.add_argument('--resume', action='store_true',
                        help="Skip models that already ran successfully in the output CSV.")
    
    args = parser.parse_args()
    
    # Load existing CSV results if resuming
    existing_results = {}
    if args.resume and args.csv and os.path.exists(args.csv):
        try:
            with open(args.csv, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = (row['model'], row['format'], row['image'])
                    existing_results[key] = {
                        'model': row['model'],
                        'format': row['format'],
                        'image': row['image'],
                        'size_mb': float(row['size_mb']) if row['size_mb'] else 0.0,
                        'avg_latency_ms': float(row['avg_latency_ms']) if row['avg_latency_ms'] else 0.0,
                        'std_latency_ms': float(row['std_latency_ms']) if row['std_latency_ms'] else 0.0,
                        'fps': float(row['fps']) if row['fps'] else 0.0,
                        'ram_mb': float(row['ram_mb']) if row['ram_mb'] else 0.0,
                        'status': row['status']
                    }
            print(f"Loaded {len(existing_results)} existing results to resume from.")
        except Exception as e:
            print(f"Warning: Could not read existing CSV to resume: {e}")

    # Locate best_weights directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    potential_dirs = []
    if args.weights_dir:
        potential_dirs.append(args.weights_dir)
    potential_dirs.extend([
        os.path.join(script_dir, '..', 'best_weights'),
        os.path.abspath(os.path.join(script_dir, '../../../best_weights')),
        os.path.abspath(os.path.join(script_dir, '../results/architecture/best_weights'))
    ])
    
    weights_dir = None
    onnx_models, tflite_models, ncnn_models = [], [], []

    for d in potential_dirs:
        if d and os.path.exists(d):
            abs_d = os.path.abspath(d)
            o_mods, tf_mods, nc_mods = discover_models_in_directory(abs_d)
            if o_mods or tf_mods or nc_mods:
                weights_dir = abs_d
                onnx_models, tflite_models, ncnn_models = o_mods, tf_mods, nc_mods
                break
            else:
                print(f"Notice: No runnable model files (.onnx, .tflite, .param/.bin) found in '{abs_d}'.")

    if not weights_dir:
        print("Error: Could not locate any directory containing runnable weight files.")
        sys.exit(1)
        
    print(f"Using weights directory: {weights_dir}")
    print(f"Discovered {len(onnx_models)} ONNX models, {len(tflite_models)} TFLite models, and {len(ncnn_models)} NCNN models.")
    
    # Setup image paths (handles directory of images or single image)
    image_paths = []
    if args.image:
        if os.path.isdir(args.image):
            valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
            image_paths = [os.path.join(args.image, f) for f in os.listdir(args.image) if f.lower().endswith(valid_exts)]
        else:
            image_paths = [args.image]
    else:
        image_paths = [None]
    results = []
    
    # --- 1. ONNX ---
    for sub, of in onnx_models:
        model_name = os.path.basename(of)
        size, _ = get_model_fallback_info(of)
        for img in image_paths:
            img_name = os.path.basename(img) if img else "dummy_input"
            key = (model_name, 'ONNX', img_name)
            if key in existing_results and existing_results[key]['status'] == 'success':
                print(f"Skipping ONNX ({sub}): {model_name} for {img_name} (using existing successful result)")
                results.append(existing_results[key])
            else:
                print(f"Running ONNX ({sub}): {model_name}...")
                res = run_onnx_inference(model_name, of, img, size, args.benchmark, args.num_runs)
                res['image'] = img_name
                results.append(res)
                        
    # --- 2. TFLite ---
    for sub, tf_file in tflite_models:
        model_name = os.path.basename(tf_file)
        size, _ = get_model_fallback_info(tf_file)
        for img in image_paths:
            img_name = os.path.basename(img) if img else "dummy_input"
            key = (model_name, 'TFLite', img_name)
            if key in existing_results and existing_results[key]['status'] == 'success':
                print(f"Skipping TFLite ({sub}): {model_name} for {img_name} (using existing successful result)")
                results.append(existing_results[key])
            else:
                print(f"Running TFLite ({sub}): {model_name}...")
                res = run_tflite_inference(model_name, tf_file, img, size, args.benchmark, args.num_runs)
                res['image'] = img_name
                results.append(res)
                        
    # --- 3. NCNN ---
    for model_name, pf, bin_file in ncnn_models:
        size, outputs = get_model_fallback_info(pf)
        for img in image_paths:
            img_name = os.path.basename(img) if img else "dummy_input"
            key = (model_name, 'NCNN', img_name)
            if key in existing_results and existing_results[key]['status'] == 'success':
                print(f"Skipping NCNN: {model_name} for {img_name} (using existing successful result)")
                results.append(existing_results[key])
            else:
                print(f"Running NCNN: {model_name}...")
                res = run_ncnn_inference(model_name, pf, bin_file, img, size, outputs, args.benchmark, args.num_runs)
                res['image'] = img_name
                results.append(res)
                        
    # Save to CSV
    if args.csv and results:
        keys = ['model', 'format', 'image', 'size_mb', 'avg_latency_ms', 'std_latency_ms', 'fps', 'ram_mb', 'status']
        with open(args.csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in results:
                writer.writerow({k: r.get(k, '') for k in keys})
        print(f"\nSaved benchmarking results to: {args.csv}")

if __name__ == '__main__':
    main()

