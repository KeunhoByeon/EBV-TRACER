import torch
import os


def get_files(base_dir, ext=None):
    file_paths = []
    for path, dir, files in os.walk(base_dir):
        for filename in files:
            if ext is not None and not filename.lower().endswith(ext.lower()):
                continue
            file_paths.append(os.path.join(path, filename))
    return file_paths

def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    ckpt_keys = list(checkpoint.keys())
    for key in ckpt_keys:
        if "classifier_.1" in key:
            checkpoint[key.replace("classifier_.1", "_fc")] = checkpoint[key]
            del checkpoint[key]
    model.load_state_dict(checkpoint)
    return model