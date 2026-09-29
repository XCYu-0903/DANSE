import os
import shutil
import json

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def load_class_paths(noise_path_json, class_define_json):  # deprecated
    with open(noise_path_json, 'r') as f:
        noise_dict = json.load(f)

    with open(class_define_json, 'r') as f:
        class_dict = json.load(f)

    result = {}
    for cls_name, keys in class_dict.items():
        paths = []
        for key in keys:
            if key not in noise_dict:
                print(f"Warning: key '{key}' not found in noise dict.")
                continue
            paths.extend(noise_dict[key])
        result[cls_name] = paths
    return result


def build_env(config, config_name, path):
    t_path = os.path.join(path, config_name)
    if config != t_path:
        os.makedirs(path, exist_ok=True)
        shutil.copyfile(config, os.path.join(path, config_name))
