import csv, yaml

def read_file(path, encoding='utf8'):
    with open(path, encoding=encoding) as file:
        return file.read()
    
def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        try:
            yaml_data = yaml.safe_load(file)
            return yaml_data
        except yaml.YAMLError as exc:
            print(exc)