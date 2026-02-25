import os
import json
from pathlib import Path
from typing import Any
import glob

def change_label(path: str = 'Organized/'):
    """Change labels in JSON files according to mapping rules."""
    
    # Define label mapping for better maintainability
    label_mapping = {
        6: 3,  # REAL -> MISLEADING
        5: 2,  # UNCLEAR -> PARTIAL
        3: 2   # MISLEADING -> PARTIAL
    }
    
    processed_count = 0
    error_count = 0
    changed_count = 0
    
    try:
        with os.scandir(path) as folders:
            for folder in folders:
                if not folder.is_dir():
                    continue
                
                print(f"Processing folder: {folder.name}")
                
                try:
                    with os.scandir(folder.path) as files:
                        for file in files:
                            if not file.is_file() or not file.name.endswith('.json'):
                                continue
                            
                            try:
                                # Read the JSON file
                                with open(file.path, 'r', encoding='utf-8') as f:
                                    data = json.load(f)
                                
                                original_label = data.get('label')
                                
                                # Check if label needs to be changed
                                if original_label in label_mapping:
                                    data['label'] = label_mapping[original_label]
                                    
                                    # Write back to file
                                    with open(file.path, 'w', encoding='utf-8') as f:
                                        json.dump(data, f, ensure_ascii=False, indent=2)
                                    
                                    changed_count += 1
                                    print(f"  Changed {file.name}: {original_label} -> {data['label']}")
                                
                                processed_count += 1
                                
                                # Progress indicator for large datasets
                                if processed_count % 100 == 0:
                                    print(f"  Processed {processed_count} files...")
                            
                            except (json.JSONDecodeError, KeyError, OSError) as e:
                                print(f"  Error processing file {file.name}: {e}")
                                error_count += 1
                                continue
                
                except OSError as e:
                    print(f"Error accessing folder {folder.name}: {e}")
                    error_count += 1
                    continue
    
    except OSError as e:
        print(f"Error accessing path {path}: {e}")
        return
    
    # Summary
    print(f"\nLabel change complete:")
    print(f"  Total processed: {processed_count} files")
    print(f"  Labels changed: {changed_count} files")
    print(f"  Errors: {error_count} files")

def add_field(feild: str = 'type', value: Any = 'article', path: str = 'Organized/'):
    with os.scandir(path) as folders:
        for folder in folders:
            if not folder.is_dir():
                continue
            
            print(f'Processing folder: {folder.name}')

            files = os.scandir(folder.path)
            for file in files:
                if not file.is_file() or not file.name.endswith('.json'):
                    continue

                with open(file.path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if feild not in data:
                        data[feild] = value
                        with open(file.path, 'w', encoding='utf-8') as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)
                            print(f"Added field '{feild}' to {file.name}")
                    else:
                        print(f"Field '{feild}' already exists in {file.name}")
        
        print(f"All files in {path} processed.")


import json
from pathlib import Path
from typing import Any

def add_field2(field: str = 'type', value: Any = 'article', path='Organized/', override: bool = False):
    path = Path(path)
    files = path.iterdir()

    for file in files:
        if not file.is_file() or not file.name.endswith('.json'):
            print(f'Skipping file: {file.name}')
            continue

        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON in {file.name}: {e}")
            continue

        if not override and field in data:
            print(f'Field "{field}" already exists in {file.name}, skipping...')
            continue

        data[field] = value

        with open(file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            print(f'Successfully added field "{field}" with value "{value}" to {file.name}')
    
    print(f"\n✅ All files in '{path}' processed.\n")



def determine_label(path = 'VFNDataset/'):
    path = Path(path)

    NAME2LABEL = {
        'Fake' : 1,
        'Misleading' : 2,
        'Real' : 3
    }

    folders = path.iterdir()
    for folder in folders:
        if not folder.is_dir():
            continue
        files = folder.iterdir()
        for file in files:
            if not file.is_file() or not file.name.endswith('.json'):
                continue

            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            data['label'] = NAME2LABEL[folder.name]

            with open(file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
                print(f'Changed label of file {file.name}')


def determine_label2(path = 'VFNDataset/'):
    path = Path(path)
    files = path.iterdir()

    for file in files:
        if not file.is_file() or not file.name.endswith('.json'):
            print(f'\nSkipping file {file}\n')
            continue

        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        data['label'] = int(data['label'])

        with open(file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            print(f'Converted label from str to int for file {file.name}')



OUTPUT_DIR = "Data/Organized"
FILES = {
    "FAKE_Social": os.path.join(OUTPUT_DIR, "FAKE_Social.jsonl"),
    "REAL_Social": os.path.join(OUTPUT_DIR, "REAL_Social.jsonl"),
    "FAKE_Article": os.path.join(OUTPUT_DIR, "FAKE_Article.jsonl"),
    "REAL_Article": os.path.join(OUTPUT_DIR, "REAL_Article.jsonl"),
}

def get_category(file_path: str):
    """
    Determine Label (Fake/Real) and Type (Social/Article) based on file path.
    Returns: (label_int, type_str)
    """
    path_str = file_path.lower()
    
    # 1. Determine Label (0: Fake, 1: Real)
    label = -1
    if 'real' in path_str:
        label = 1
    elif 'fake' in path_str or 'misleading' in path_str:
        label = 0
    
    # 2. Determine Type (Social vs Article)
    # Default to Article unless 'social' is found in path
    dtype = "Article"
    if 'social' in path_str:
        dtype = "Social"
    elif 'article' in path_str:
        dtype = "Article"
        
    return label, dtype

def migrate():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Open file handles for writing
    file_handles = {}
    for key, path in FILES.items():
        file_handles[key] = open(path, 'w', encoding='utf-8')

    stats = {key: 0 for key in FILES}
    skipped = 0

    print(f"🚀 Scanning only 'Organized/' folder...")
    
    # STRICTLY glob only the Organized folder as requested
    input_files = glob.glob("Organized/**/*.json", recursive=True)
    input_files = list(set(input_files))

    print(f"📂 Found {len(input_files)} legacy files.")

    for file_path in input_files:
        # Skip package/config files
        if not file_path.endswith(".json") or "package" in file_path:
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f_in:
                try:
                    data = json.load(f_in)
                except:
                    skipped += 1
                    print(f"Skipped v1")
                    continue

                # Extract Text
                text = data.get("text", "")
                if not text:
                    text = data.get("paragraphs", "")
                if not text:
                    text = data.get("maintext", "")
                if not text: 
                    # Handle case where content is a dict
                    c = data.get("content", "")
                    if isinstance(c, dict): text = c.get("text", "")
                    else: text = str(c)
                
                # Validation: Skip empty/short text
                if not text or len(text) < 10:
                    print(file_path)
                    skipped += 1
                    continue

                # --- CLASSIFICATION ---
                label, dtype = get_category(file_path)
                
                # Fallback: Check internal JSON label if path is unclear
                if label == -1:
                    raw_label = str(data.get("label", "")).lower()
                    if raw_label in ["fake", "0", "false", "misleading"]:
                        label = 0
                    elif raw_label in ["real", "1", "true"]:
                        label = 1

                # If still unknown, skip
                if label == -1:
                    skipped += 1
                    # print(file_path)
                    continue
                
                # Determine Output Key
                label_str = "FAKE" if label == 0 else "REAL"
                key = f"{label_str}_{dtype}"

                # Create Standard Object
                safe_id = f"LEGACY_{os.path.basename(file_path).replace('.json', '')}"
                std_obj = {
                    "id": safe_id,
                    "text": text,
                    "source": data.get("source_domain", "legacy"),
                    "label": label,
                    "date": data.get("date_published"),
                    "type": dtype.lower(),
                    "is_legacy": True
                }

                # Write to specific file
                if key in file_handles:
                    file_handles[key].write(json.dumps(std_obj, ensure_ascii=False) + "\n")
                    stats[key] += 1
                else:
                    skipped += 1

        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            skipped += 1

    # Close handles
    for f in file_handles.values():
        f.close()

    print("-" * 30)
    print("✅ MIGRATION COMPLETE!")
    for key, count in stats.items():
        print(f"  📄 {key}: {count} lines -> {FILES[key]}")
    print(f"  🗑️ Skipped: {skipped} files")
if __name__ == '__main__':
    migrate()