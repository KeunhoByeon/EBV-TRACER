import os


def get_files(base_dir, ext=None):
    file_paths = []
    for path, dir, files in os.walk(base_dir):
        for filename in files:
            if ext is not None and not filename.lower().endswith(ext.lower()):
                continue
            file_paths.append(os.path.join(path, filename))
    return file_paths
