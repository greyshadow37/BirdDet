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
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size)
        img_np = np.array(img, dtype=np.float32) / 255.0
        # HWC to CHW (ncnn/onnx/pytorch expect CHW)
        img_np = img_np.transpose((2, 0, 1))
        return np.expand_dims(img_np, axis=0)
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None

def run_ncnn_inference(model_name, param_path, bin_path, image_path, fallback_size, outputs_list, benchmark=False, num_runs=100):
    """Runs inference using NCNN backend."""
    import ncnn
    total_size = (os.path.getsize(param_path) + os.path.getsize(bin_path)) / (1024 * 1024)
    
    metrics = {
        'model': model_name,
        'format': 'NCNN',
        'size_mb': round(total_size, 3),
        'avg_latency_ms': 0.0,
        'std_latency_ms': 0.0,
        'fps': 0.0,
        'ram_mb': 0.0,
        'status': 'success'
    }
    
    try:
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
    import onnxruntime as ort
    total_size = os.path.getsize(model_path) / (1024 * 1024)
    
    metrics = {
        'model': model_name,
        'format': 'ONNX',
        'size_mb': round(total_size, 3),
        'avg_latency_ms': 0.0,
        'std_latency_ms': 0.0,
        'fps': 0.0,
        'ram_mb': 0.0,
        'status': 'success'
    }
    
    try:
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
    import tensorflow as tf
    total_size = os.path.getsize(model_path) / (1024 * 1024)
    
    metrics = {
        'model': model_name,
        'format': 'TFLite',
        'size_mb': round(total_size, 3),
        'avg_latency_ms': 0.0,
        'std_latency_ms': 0.0,
        'fps': 0.0,
        'ram_mb': 0.0,
        'status': 'success'
    }
    
    try:
        # Disable XNNPACK delegate if running NanoDet INT8 to prevent Windows crash
        if 'nanodet' in model_name.lower() and 'int8' in model_name.lower():
            interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=4)
        else:
            interpreter = tf.lite.Interpreter(model_path=model_path)
            
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
        if input_dtype == np.uint8:
            tflite_input = (tflite_input * 255.0).astype(np.uint8)
        elif input_dtype == np.int8:
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

def run_pytorch_inference(model_name, model_path, image_path, fallback_size, benchmark=False, num_runs=100):
    """Runs inference using PyTorch backend."""
    import torch
    total_size = os.path.getsize(model_path) / (1024 * 1024)
    
    metrics = {
        'model': model_name,
        'format': 'PyTorch',
        'size_mb': round(total_size, 3),
        'avg_latency_ms': 0.0,
        'std_latency_ms': 0.0,
        'fps': 0.0,
        'ram_mb': 0.0,
        'status': 'success'
    }
    
    try:
        # Preprocess input image using fallback size
        if image_path:
            input_tensor = preprocess_image(image_path, fallback_size)
        else:
            input_tensor = np.random.rand(1, 3, fallback_size[1], fallback_size[0]).astype(np.float32)
            
        if input_tensor is None:
            raise ValueError("Input tensor preprocessing failed.")

        torch_tensor = torch.from_numpy(input_tensor).float()
        
        if 'nanodet' in model_name.lower():
            # Load NanoDet model
            script_dir = os.path.dirname(os.path.abspath(__file__))
            nanodet_dir = os.path.abspath(os.path.join(script_dir, '..', '..', 'nanodet'))
            if nanodet_dir not in sys.path:
                sys.path.insert(0, nanodet_dir)
                
            from nanodet.model.arch import build_model
            from nanodet.util.config import cfg, load_config
            
            # Find train_cfg.yml
            potential_cfgs = glob.glob(os.path.join(script_dir, '..', 'results', 'train', 'NanoDet', '**', 'train_cfg.yml'), recursive=True)
            if not potential_cfgs:
                raise FileNotFoundError("Could not find train_cfg.yml for NanoDet")
            
            load_config(cfg, potential_cfgs[0])
            
            # Update config input size to match fallback_size if needed
            cfg.defrost()
            cfg.data.val.input_size = [fallback_size[1], fallback_size[0]]
            cfg.freeze()
            
            model = build_model(cfg.model)
            checkpoint = torch.load(model_path, map_location='cpu')
            state_dict = checkpoint.get('state_dict', checkpoint)
            if any(k.startswith('model.') for k in state_dict.keys()):
                state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            
            # Warmup
            with torch.no_grad():
                model(torch_tensor)
                
            if benchmark:
                latencies = []
                mem_start = get_peak_memory()
                with torch.no_grad():
                    for _ in range(num_runs):
                        t0 = time.perf_counter()
                        model(torch_tensor)
                        latencies.append((time.perf_counter() - t0) * 1000.0)
                mem_end = get_peak_memory()
                
                avg_lat = np.mean(latencies)
                metrics['avg_latency_ms'] = round(avg_lat, 3)
                metrics['std_latency_ms'] = round(np.std(latencies), 3)
                metrics['fps'] = round(1000.0 / avg_lat, 2)
                metrics['ram_mb'] = round(mem_end, 2)
        else:
            # YOLO / RT-DETR
            from ultralytics import YOLO
            model = YOLO(model_path)
            
            # Use numpy input for ultralytics YOLO predict to keep it consistent
            raw_img = np.transpose(input_tensor[0], (1, 2, 0)) # CHW to HWC
            
            # Warmup
            model.predict(raw_img, imgsz=fallback_size[0], device='cpu', verbose=False)
            
            if benchmark:
                latencies = []
                mem_start = get_peak_memory()
                for _ in range(num_runs):
                    t0 = time.perf_counter()
                    model.predict(raw_img, imgsz=fallback_size[0], device='cpu', verbose=False)
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
    
    args = parser.parse_args()
    
    # Locate best_weights directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    potential_dirs = [
        args.weights_dir,
        os.path.join(script_dir, '..', 'best_weights'),
        os.path.abspath(os.path.join(script_dir, '../../../best_weights'))
    ]
    
    weights_dir = None
    for d in potential_dirs:
        if d and os.path.exists(d):
            weights_dir = os.path.abspath(d)
            break
            
    if not weights_dir:
        print("Error: Could not locate 'best_weights' directory.")
        sys.exit(1)
        
    print(f"Using weights directory: {weights_dir}")
    
    # Setup image path
    image_paths = [args.image] if args.image else [None]
    results = []
    
    # --- 1. PyTorch (.pt, .pth) ---
    pt_dir = os.path.join(weights_dir, 'pt_or_pth')
    if os.path.exists(pt_dir):
        pt_files = glob.glob(os.path.join(pt_dir, '*.pt')) + glob.glob(os.path.join(pt_dir, '*.pth'))
        for pf in pt_files:
            model_name = os.path.basename(pf)
            size, _ = get_model_fallback_info(pf)
            for img in image_paths:
                img_name = os.path.basename(img) if img else "dummy_input"
                print(f"Running PyTorch: {model_name}...")
                res = run_pytorch_inference(model_name, pf, img, size, args.benchmark, args.num_runs)
                res['image'] = img_name
                results.append(res)
                    
    # --- 2. ONNX (Standard & INT8) ---
    for sub in ['onnx', 'onnx_int8']:
        o_dir = os.path.join(weights_dir, sub)
        if os.path.exists(o_dir):
            onnx_files = glob.glob(os.path.join(o_dir, '*.onnx'))
            for of in onnx_files:
                model_name = os.path.basename(of)
                size, _ = get_model_fallback_info(of)
                for img in image_paths:
                    img_name = os.path.basename(img) if img else "dummy_input"
                    print(f"Running ONNX ({sub}): {model_name}...")
                    res = run_onnx_inference(model_name, of, img, size, args.benchmark, args.num_runs)
                    res['image'] = img_name
                    results.append(res)
                        
    # --- 3. TFLite (Standard & INT8) ---
    for sub in ['tflite', 'tflite_int8']:
        tf_dir = os.path.join(weights_dir, sub)
        if os.path.exists(tf_dir):
            tflite_files = glob.glob(os.path.join(tf_dir, '**', '*.tflite'), recursive=True)
            for f in os.listdir(tf_dir):
                full_p = os.path.join(tf_dir, f)
                if os.path.isfile(full_p) and 'tflite' in f.lower() and not f.endswith('.tflite'):
                    tflite_files.append(full_p)
                    
            for tf_file in sorted(list(set(tflite_files))):
                model_name = os.path.basename(tf_file)
                size, _ = get_model_fallback_info(tf_file)
                for img in image_paths:
                    img_name = os.path.basename(img) if img else "dummy_input"
                    print(f"Running TFLite ({sub}): {model_name}...")
                    res = run_tflite_inference(model_name, tf_file, img, size, args.benchmark, args.num_runs)
                    res['image'] = img_name
                    results.append(res)
                        
    # --- 4. NCNN ---
    ncnn_dir = os.path.join(weights_dir, 'ncnn')
    if os.path.exists(ncnn_dir):
        subdirs = [os.path.join(ncnn_dir, d) for d in os.listdir(ncnn_dir) if os.path.isdir(os.path.join(ncnn_dir, d))]
        for sd in subdirs:
            param_files = glob.glob(os.path.join(sd, '*.param'))
            for pf in param_files:
                model_name = os.path.basename(sd)
                bin_file = pf.replace('.param', '.bin')
                if not os.path.exists(bin_file):
                    continue
                size, outputs = get_model_fallback_info(sd)
                for img in image_paths:
                    img_name = os.path.basename(img) if img else "dummy_input"
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
