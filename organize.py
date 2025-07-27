import os
import json
import shutil
from pathlib import Path
from typing import Dict

LABLE2NAME = {
    1 :'FAKE',
    2: 'MISLEADING',
    3: 'REAL'
}

TYPES = {
    'Article' : 0,
    'Social' : 0 
}

def organize_files(sourceFolder: str = 'Data/', desFolder: str = 'Organized/'):
    """Organize JSON files by label into separate folders."""
    print(f"Organizing files from {sourceFolder} to {desFolder}")
    
    # Create all destination folders at once
    file_counts: Dict[str, Dict[str,int]] = {}
    processed_count = 0
    skipped_count = 0
    error_count = 0

    for label_name in LABLE2NAME.values():
        folder_path = Path(os.path.join(desFolder, label_name))
        folder_path.mkdir(parents=True, exist_ok=True)
        for type in TYPES.keys():
            type_folder = folder_path / type
            type_folder.mkdir(parents=True, exist_ok=True)

            file_counts[label_name] = {type : sum(1 for entry in os.scandir(type_folder) if entry.is_file() and entry.name.endswith('.json'))}

            with open(type_folder / 'Placeholder.txt', 'w') as f:
                f.write("placeholder")
    
    # Process files
    try:
        with os.scandir(sourceFolder) as files:
            for file in files:
                if not file.is_file() or not file.name.endswith('.json'):
                    skipped_count += 1
                    print(f"\nSkipping non-JSON file: {file.name}\n")
                    continue
                
                try:
                    with open(file.path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        label = data.get('label')
                        type = data.get('type')
                    
                        # Validate label
                        if not isinstance(label, int) or label not in LABLE2NAME:
                            print(f"\nInvalid label for file {file.name}: {label}\n")
                            error_count += 1
                            continue
                        elif not isinstance(type, str) or type not in {'article', 'social'}:
                            print(f'\nInvalid type for file {file.name}: {type}\n')
                            error_count += 1
                            continue
                    
                    # Generate new file path
                    type = type.capitalize()
                    type_folder = Path(os.path.join(desFolder, LABLE2NAME[label], type))
                    file_count = file_counts[LABLE2NAME[label]].get(type, 0)
                    file_counts[LABLE2NAME[label]][type] = file_count + 1
                    file_name = f'FND_{type}_{file_count + 1}.json'
                    new_path = type_folder / file_name
                    
                    # Move file
                    shutil.move(file.path, new_path)
                    processed_count += 1
                    print(f"Moved {file.name} to {new_path}")
                
                except (json.JSONDecodeError, KeyError, OSError) as e:
                    print(f"Error processing file {file.name}: {e}")
                    error_count += 1
                    continue
    
    except OSError as e:
        print(f"Error accessing source folder {sourceFolder}: {e}")
        return
    
    # Summary
    print(f"\nProcessing complete:")
    print(f"  Processed: {processed_count} files")
    print(f"  Skipped: {skipped_count} files")
    print(f"  Errors: {error_count} files")

            

if __name__ == '__main__':
    organize_files(sourceFolder='Socials/', desFolder='Organized/')
    print("Files organized successfully.")