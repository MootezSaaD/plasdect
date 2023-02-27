import glob, os

def read_file(path, encoding='utf8'):
    with open(path, encoding=encoding) as file:
        return file.read()
    
def read_file_lines(path, encoding='utf8'):
    with open(path, encoding=encoding) as file:
        return file.readlines()
    
def load_files_by_ext(path, extension='*', encoding='utf8'):
    return list(glob.iglob(os.path.join(path,'*', f'*.{extension}'), recursive=True))

def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)

def batch_delete(files_batch):
    for file in files_batch:
        os.remove(file)
        print(f'deleted {file}')