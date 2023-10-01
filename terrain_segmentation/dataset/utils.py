import os

def get_paths(path):
    file_paths = []
    for file in sorted(os.listdir(path)):
        file_paths.append(path / file)
    return file_paths
