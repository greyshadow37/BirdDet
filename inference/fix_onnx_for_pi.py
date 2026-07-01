"""
fix_onnx_for_pi.py
-------------------
Standalone script to fix ALL known ONNX compatibility issues for the
Raspberry Pi's older ONNX Runtime (1.11-1.14).

Handles:
  1. Missing opset import entries       (pnnx.onnx)
  2. Missing / zero IR version          (pnnx.onnx)
  3. Reshape nodes: legacy 'shape' attr (pnnxsim.onnx)
  4. Unsupported Gelu operator          (RT-DETR_self_contained.onnx)
  5. ReduceMax 2-input -> 1-input+attr  (RT-DETR_self_contained.onnx)
  6. IR version > 9
  7. Opset version > 17
  8. Legacy Split 'num_outputs' attribute

Run from the BirdDet root:
  python inference/fix_onnx_for_pi.py --onnx_dir best_weights/onnx
"""

import os
import argparse
import numpy as np

try:
    import onnx
    from onnx import helper, TensorProto, numpy_helper
except ImportError:
    print("ERROR: 'onnx' package not installed. Run:  pip install onnx")
    exit(1)


# ───────────────────────── individual fixers ─────────────────────────

def fix_missing_opset(model):
    """Add a default ai.onnx opset import if none exists."""
    has_default = any(
        (op.domain == "" or op.domain == "ai.onnx") for op in model.opset_import
    )
    if not has_default:
        print("  [FIX] Adding missing default opset import (ai.onnx version 17)")
        opset = onnx.OperatorSetIdProto()
        opset.domain = ""
        opset.version = 17
        model.opset_import.append(opset)
        return True
    return False


def fix_ir_version(model, target_ir=9):
    """Set IR version to target_ir if it is 0 (missing) or above target_ir."""
    if model.ir_version == 0:
        print(f"  [FIX] Setting missing IR version -> {target_ir}")
        model.ir_version = target_ir
        return True
    if model.ir_version > target_ir:
        print(f"  [FIX] Downgrading IR version {model.ir_version} -> {target_ir}")
        model.ir_version = target_ir
        return True
    return False


def fix_opset_version(model, max_opset=17):
    """Cap opset version at max_opset."""
    modified = False
    for opset in model.opset_import:
        if (opset.domain == "" or opset.domain == "ai.onnx") and opset.version > max_opset:
            print(f"  [FIX] Downgrading opset {opset.version} -> {max_opset}")
            opset.version = max_opset
            modified = True
    return modified


def fix_split_nodes(model):
    """Remove legacy 'num_outputs' attribute from Split nodes."""
    modified = False
    for node in model.graph.node:
        if node.op_type == "Split":
            for attr in list(node.attribute):
                if attr.name == "num_outputs":
                    print(f"  [FIX] Removing 'num_outputs' from Split node: {node.name}")
                    node.attribute.remove(attr)
                    modified = True
    return modified


def fix_reshape_nodes(model):
    """
    Fix Reshape nodes that have only 1 input (missing shape tensor).

    The pnnxsim exporter produces Reshape nodes with:
      - Only 1 input (data)
      - A legacy 'shape' attribute containing the target shape

    Older ONNX Runtime expects Reshape (opset >= 5) to have exactly 2 inputs:
      input[0] = data, input[1] = shape tensor (int64)
    and NO 'shape' attribute.

    This function:
      1. Reads the target shape from the 'shape' attribute
      2. Creates an initializer tensor with that shape
      3. Adds it as the 2nd input
      4. Removes the 'shape' attribute
    """
    modified = False
    graph = model.graph

    # Build a map of tensor name -> shape from value_info and outputs (fallback)
    shape_map = {}
    for vi in list(graph.value_info) + list(graph.output):
        if vi.type.HasField("tensor_type") and vi.type.tensor_type.HasField("shape"):
            dims = []
            for d in vi.type.tensor_type.shape.dim:
                dims.append(d.dim_value if d.dim_value > 0 else -1)
            shape_map[vi.name] = dims

    for node in graph.node:
        if node.op_type != "Reshape":
            continue

        # Check if there's a legacy 'shape' attribute to remove
        shape_attr = None
        for attr in node.attribute:
            if attr.name == "shape":
                shape_attr = attr
                break

        needs_second_input = len(node.input) < 2 or (len(node.input) == 2 and node.input[1] == "")

        if shape_attr is not None or needs_second_input:
            # Determine the target shape
            target_shape = None

            # Priority 1: from the 'shape' attribute
            if shape_attr is not None:
                target_shape = list(shape_attr.ints)
                print(f"  [FIX] Reshape {node.name}: using shape from attribute = {target_shape}")
                node.attribute.remove(shape_attr)

            # Priority 2: from output value_info
            if target_shape is None:
                output_name = node.output[0] if node.output else None
                if output_name and output_name in shape_map:
                    target_shape = shape_map[output_name]
                    print(f"  [FIX] Reshape {node.name}: using shape from output info = {target_shape}")

            # Priority 3: fallback
            if target_shape is None:
                target_shape = [0, -1]
                print(f"  [FIX] Reshape {node.name}: using fallback shape = {target_shape}")

            # Remove any other Reshape-incompatible attributes
            for attr in list(node.attribute):
                if attr.name == "allowzero":
                    pass  # this is valid
                else:
                    print(f"  [FIX] Reshape {node.name}: removing attribute '{attr.name}'")
                    node.attribute.remove(attr)

            # Create shape initializer and add as 2nd input
            if needs_second_input:
                shape_name = f"{node.name}_shape_fix"
                shape_tensor = numpy_helper.from_array(
                    np.array(target_shape, dtype=np.int64), name=shape_name
                )
                graph.initializer.append(shape_tensor)
                if len(node.input) < 2:
                    node.input.append(shape_name)
                else:
                    node.input[1] = shape_name

            modified = True

    return modified


def fix_reducemax_nodes(model):
    """
    Convert ReduceMax from 2-input format (opset 18+) to 1-input + axes attribute
    format (opset <= 17) for older ONNX Runtime compatibility.

    In opset 18+: ReduceMax(data, axes_tensor) -> output
    In opset <= 17: ReduceMax(data, axes=[...] attribute) -> output
    """
    graph = model.graph
    modified = False

    # Build initializer lookup: name -> numpy array
    init_map = {}
    for init in graph.initializer:
        try:
            init_map[init.name] = numpy_helper.to_array(init)
        except Exception:
            pass

    for node in graph.node:
        if node.op_type in ("ReduceMax", "ReduceMin", "ReduceMean", "ReduceSum",
                            "ReduceProd", "ReduceL1", "ReduceL2",
                            "ReduceLogSum", "ReduceLogSumExp", "ReduceSumSquare"):
            if len(node.input) >= 2 and node.input[1] != "":
                axes_input_name = node.input[1]

                # Try to read the axes value from initializers
                if axes_input_name in init_map:
                    axes_val = init_map[axes_input_name].flatten().tolist()
                    axes_val = [int(a) for a in axes_val]
                else:
                    # Try to find it as a Constant node output
                    axes_val = None
                    for other_node in graph.node:
                        if other_node.op_type == "Constant" and other_node.output[0] == axes_input_name:
                            for attr in other_node.attribute:
                                if attr.name == "value":
                                    t = numpy_helper.to_array(attr.t)
                                    axes_val = t.flatten().tolist()
                                    axes_val = [int(a) for a in axes_val]
                                    break
                            break

                if axes_val is not None:
                    print(f"  [FIX] {node.op_type} {node.name}: converting 2-input -> attribute axes={axes_val}")

                    # Remove the 2nd input (axes tensor)
                    del node.input[1:]

                    # Check if 'axes' attribute already exists
                    has_axes = any(a.name == "axes" for a in node.attribute)
                    if not has_axes:
                        node.attribute.append(helper.make_attribute("axes", axes_val))

                    modified = True
                else:
                    print(f"  [WARN] {node.op_type} {node.name}: could not resolve axes tensor '{axes_input_name}'")

    return modified


def decompose_gelu(model):
    """
    Replace Gelu nodes with the sigmoid approximation:
      gelu(x) ≈ x * sigmoid(1.702 * x)
    """
    graph = model.graph
    nodes_to_remove = []
    nodes_to_add = []
    modified = False

    for node in graph.node:
        if node.op_type == "Gelu":
            print(f"  [FIX] Decomposing Gelu node: {node.name}")
            inp = node.input[0]
            out = node.output[0]
            prefix = node.name or f"gelu_fix_{id(node)}"

            # Constant 1.702
            c_name = f"{prefix}_coeff"
            c_tensor = numpy_helper.from_array(
                np.array([1.702], dtype=np.float32), name=c_name
            )
            graph.initializer.append(c_tensor)

            # Mul: scaled = 1.702 * x
            mul_node = helper.make_node("Mul", [inp, c_name], [f"{prefix}_scaled"], name=f"{prefix}_mul")

            # Sigmoid: sig = sigmoid(scaled)
            sig_node = helper.make_node("Sigmoid", [f"{prefix}_scaled"], [f"{prefix}_sig"], name=f"{prefix}_sigmoid")

            # Mul: out = x * sig
            out_node = helper.make_node("Mul", [inp, f"{prefix}_sig"], [out], name=f"{prefix}_out_mul")

            nodes_to_remove.append(node)
            nodes_to_add.extend([mul_node, sig_node, out_node])
            modified = True

    for n in nodes_to_remove:
        graph.node.remove(n)
    for n in nodes_to_add:
        graph.node.append(n)

    return modified


def fix_resize_nodes(model):
    """Remove unsupported attributes from Resize nodes."""
    modified = False
    allowed_attrs = {"coordinate_transformation_mode", "cubic_coeff_a", "exclude_outside",
                     "extrapolation_value", "mode", "nearest_mode"}
    for node in model.graph.node:
        if node.op_type == "Resize":
            for attr in list(node.attribute):
                if attr.name not in allowed_attrs:
                    print(f"  [FIX] Removing unsupported Resize attribute '{attr.name}' from: {node.name}")
                    node.attribute.remove(attr)
                    modified = True
    return modified


# ───────────────────────── main entry point ─────────────────────────

def fix_onnx_model(onnx_path):
    """Apply all fixes to a single ONNX model file."""
    print(f"\n--- Processing: {os.path.basename(onnx_path)} ---")

    try:
        model = onnx.load(onnx_path)
    except Exception as e:
        print(f"  ERROR: Could not load {onnx_path}: {e}")
        return False

    modified = False
    modified |= fix_missing_opset(model)
    modified |= fix_ir_version(model)
    modified |= fix_opset_version(model)
    modified |= fix_split_nodes(model)
    modified |= fix_reshape_nodes(model)
    modified |= fix_reducemax_nodes(model)
    modified |= decompose_gelu(model)
    modified |= fix_resize_nodes(model)

    if modified:
        onnx.save(model, onnx_path)
        print(f"  SAVED: {onnx_path}")
    else:
        print(f"  Already compatible, no changes needed.")

    return modified


def main():
    parser = argparse.ArgumentParser(
        description="Fix ALL ONNX models for Raspberry Pi compatibility"
    )
    parser.add_argument(
        "--onnx_dir",
        type=str,
        default="best_weights/onnx",
        help="Directory containing ONNX model files",
    )
    args = parser.parse_args()

    onnx_dir = args.onnx_dir
    if not os.path.isdir(onnx_dir):
        print(f"ERROR: Directory not found: {onnx_dir}")
        return

    onnx_files = [f for f in os.listdir(onnx_dir) if f.endswith(".onnx")]
    if not onnx_files:
        print(f"No .onnx files found in {onnx_dir}")
        return

    print(f"Found {len(onnx_files)} ONNX file(s) in {onnx_dir}")
    print("=" * 60)

    fixed_count = 0
    for f in sorted(onnx_files):
        path = os.path.join(onnx_dir, f)
        if fix_onnx_model(path):
            fixed_count += 1

    print("\n" + "=" * 60)
    print(f"Done. Fixed {fixed_count}/{len(onnx_files)} model(s).")
    print("\nNext steps:")
    print("  1. Copy the 3 fixed ONNX files to the Pi:")
    print(f"     scp {onnx_dir}/RT-DETR_self_contained.onnx ayan@192.168.29.219:~/bird_study/best_weights/onnx/")
    print(f"     scp {onnx_dir}/nanodet_model_best_self_contained.pnnx.onnx ayan@192.168.29.219:~/bird_study/best_weights/onnx/")
    print(f"     scp {onnx_dir}/nanodet_model_best_self_contained.pnnxsim.onnx ayan@192.168.29.219:~/bird_study/best_weights/onnx/")
    print("  2. Re-run the benchmark on the Pi:")
    print("     ./run_pi_benchmarks.sh")


if __name__ == "__main__":
    main()
