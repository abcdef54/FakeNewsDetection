import os
import json
import shutil

LABLE2NAME = {
    1 :'FAKE',
    2: 'MISLEADING',
    3: 'REAL'
}

def organize_files(sourceFolder: str = 'Data/', desFolder: str = 'Organized/'):
    """Organize JSON files by label into separate folders."""
    print(f"Organizing files from {sourceFolder} to {desFolder}")
    
    # Create all destination folders at once
    for label_name in LABLE2NAME.values():
        folder_path = os.path.join(desFolder, label_name)
        os.makedirs(folder_path, exist_ok=True)
    
    # Cache file counts for each label folder to avoid repeated scanning
    file_counts = {}
    for label, label_name in LABLE2NAME.items():
        label_folder = os.path.join(desFolder, label_name)
        file_counts[label] = sum(1 for entry in os.scandir(label_folder) if entry.is_file())
    
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
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
                    
                    # Validate label
                    if not isinstance(label, int) or label not in LABLE2NAME:
                        print(f"\nInvalid label for file {file.name}: {label}\n")
                        error_count += 1
                        continue
                    
                    # Generate new file path
                    label_folder = os.path.join(desFolder, LABLE2NAME[label])
                    file_counts[label] += 1
                    file_name = f'FakeNewsDetection_{file_counts[label]}.json'
                    new_path = os.path.join(label_folder, file_name)
                    
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
    organize_files(sourceFolder='Data/', desFolder='Organized/')
    print("Files organized successfully.")