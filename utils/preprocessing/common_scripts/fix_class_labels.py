import os
import argparse
from pathlib import Path
import yaml

def load_class_names(data_yaml_path):
    """Load class names from data.yaml"""
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    names = data.get('names', [])
    if not names:
        raise ValueError("No 'names' field found in data.yaml")
    return names

def get_class_id_from_folder_name(folder_name, class_names):
    """Map folder name to class ID using class_names list"""
    try:
        return class_names.index(folder_name)
    except ValueError:
        raise ValueError(f"Folder name '{folder_name}' not found in class names: {class_names}")

def process_labels_in_folder(class_folder_path, class_id, class_names):
    """
    Process all .txt files in train/val/test subdirectories under class_folder_path/labels/
    Replace ALL class IDs in labels with the correct class_id.
    """
    labels_base = class_folder_path / "labels"
    subsets = ["train", "val", "test"]

    for subset in subsets:
        subset_label_dir = labels_base / subset
        if not subset_label_dir.exists():
            print(f"⚠️  Skipping non-existent directory: {subset_label_dir}")
            continue

        txt_files = list(subset_label_dir.glob("*.txt"))
        if not txt_files:
            print(f"ℹ️  No label files found in {subset_label_dir}")
            continue

        print(f"🔧 Processing {len(txt_files)} label files in {subset_label_dir} → class {class_id} ({class_names[class_id]})")

        for txt_file in txt_files:
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                # Backup original
                backup_path = txt_file.with_suffix(txt_file.suffix + ".bak")
                if not backup_path.exists():
                    with open(backup_path, 'w', encoding='utf-8') as bf:
                        bf.writelines(lines)

                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        # Keep malformed lines untouched (but warn)
                        new_lines.append(line)
                        continue
                    # Replace class ID with correct one
                    parts[0] = str(class_id)
                    new_lines.append(" ".join(parts) + "\n")

                # Write updated file
                with open(txt_file, 'w', encoding='utf-8') as f:
                    f.writelines(new_lines)

            except Exception as e:
                print(f"❌ Error processing {txt_file}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Fix YOLO class IDs in label files based on parent folder name."
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Root directory containing class subfolders (e.g., asian-green-bee-eater/, cattle-egret/, etc.)'
    )
    parser.add_argument(
        '--data_yaml',
        type=str,
        default=None,
        help='Path to data.yaml (default: data_dir/data.yaml)'
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_yaml_path = Path(args.data_yaml) if args.data_yaml else data_dir / "data.yaml"

    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found at {data_yaml_path}. Provide path with --data_yaml")

    # Load class names
    class_names = load_class_names(data_yaml_path)
    nc = len(class_names)
    print(f"✅ Loaded {nc} classes: {class_names}")

    # Get all subdirectories under data_dir (should be class folders)
    class_folders = [f for f in data_dir.iterdir() if f.is_dir()]
    valid_class_folders = []

    for folder in class_folders:
        if folder.name in class_names:
            valid_class_folders.append(folder)
        else:
            print(f"⚠️  Skipping unknown folder: {folder.name} (not in data.yaml names)")

    if not valid_class_folders:
        raise ValueError("No valid class folders found matching data.yaml names.")

    print(f"🔍 Found {len(valid_class_folders)} valid class folders to process.")

    # Process each class folder
    for class_folder in valid_class_folders:
        class_id = get_class_id_from_folder_name(class_folder.name, class_names)
        print(f"\n➡️  Processing class '{class_folder.name}' → ID {class_id}")
        process_labels_in_folder(class_folder, class_id, class_names)

    print("\n🎉 All label files have been updated!")
    print("💡 Original files are backed up with .bak extension.")

if __name__ == "__main__":
    main()