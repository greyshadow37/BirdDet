import os
import sys
import shutil
import subprocess
import argparse

def export_nanodet_to_ncnn(pth_path, config_path, output_dir):
    print(f"\n=== Converting NanoDet model to NCNN: {pth_path} ===")
    try:
        # First export to ONNX
        nanodet_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'nanodet'))
        sys.path.insert(0, nanodet_dir)
        
        onnx_path = pth_path.replace('.pth', '.onnx')
        export_script = os.path.join(nanodet_dir, 'tools', 'export_onnx.py')
        
        print("Exporting PyTorch model to ONNX first...")
        subprocess.run([
            sys.executable,
            export_script,
            '--cfg_path', config_path,
            '--model_path', pth_path,
            '--out_path', onnx_path
        ], check=True)
        
        import onnx
        print("Loading ONNX model and saving as self-contained...")
        model = onnx.load(onnx_path)
        self_contained_onnx = onnx_path.replace('.onnx', '_self_contained.onnx')
        onnx.save(model, self_contained_onnx)

        os.makedirs(output_dir, exist_ok=True)

        print("Running pnnx conversion...")
        cmd = ['pnnx', self_contained_onnx]
        subprocess.run(cmd, cwd=output_dir, check=True)

        # Clean up temporary ONNX files
        if os.path.exists(self_contained_onnx):
            os.remove(self_contained_onnx)
        if os.path.exists(onnx_path):
            os.remove(onnx_path)

        print(f"Successfully converted NanoDet and saved to: {output_dir}")
    except Exception as e:
        print(f"Error exporting NanoDet model to NCNN: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    args = parser.parse_args()
    
    export_nanodet_to_ncnn(args.model_path, args.config_path, args.out_path)

if __name__ == "__main__":
    main()
