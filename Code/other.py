import os
import json
from pathlib import Path
from typing import Any

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

if __name__ == '__main__':
    add_field2('label', 1, 'Socials/')
    add_field2('type', 'social', 'Socials/')