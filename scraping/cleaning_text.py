import os
import shutil
import tempfile

def clean_file_in_place(filepath):
    """
    Clean a single text file in place:
    - Replace header blocks (--- or Page X) with '#'
    - Remove empty lines
    """
    temp_fd, temp_path = tempfile.mkstemp()
    
    try:
        with os.fdopen(temp_fd, 'w', encoding='utf-8') as temp_file:
            with open(filepath, 'r', encoding='utf-8') as original_file:
                is_header_block = False
                for line in original_file:
                    stripped_line = line.strip()
                    
                    if '---' in stripped_line or stripped_line.startswith('Page '):
                        if not is_header_block:
                            temp_file.write('#\n')
                            is_header_block = True
                    elif stripped_line:
                        temp_file.write(line)
                        is_header_block = False
        
        shutil.move(temp_path, filepath)
        return True

    except Exception as e:
        print(f"Error processing {os.path.basename(filepath)}: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False

def clean_directory(directory_path):
    """
    Clean all .txt files in a directory
    """
    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found at '{directory_path}'")
        return

    files_found = 0
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            files_found += 1
            full_path = os.path.join(directory_path, filename)
            clean_file_in_place(full_path)
    
    if files_found == 0:
        print("No .txt files were found to clean.")

def run_cleaning():
    """
    Determine the data folder dynamically and clean all text files
    """
    # Assume data folder is inside project: project_root/data/texts
    project_root = os.path.dirname(os.path.abspath(__file__))  # folder of this script
    target_directory = os.path.join(project_root, 'data', 'texts')

    print(f"Cleaning text files in: {target_directory}")
    clean_directory(target_directory)
