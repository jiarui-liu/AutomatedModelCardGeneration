import json
from typing import Any
import uuid
import os
import pandas as pd
import csv
import numpy as np

url_regex = r"https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9@:%_\+.~#?&//=]*)"

github_url_regex = r"(?:(?:git|ssh|http(?:s)?)|(?:git@[\w\.]+))(?::(?://)?)github\.com\/(?:[\w\.@\:/\-~]+)(?:/)?"

# https://info.arxiv.org/help/arxiv_identifier_for_services.html
arxiv_id_regex = r"(?:[0-9]{4}\.[0-9]{4,5})|(?:[a-z\-]+\/[0-9]{7})"

md_code_block_regex = r"```[^\S\r\n]*[a-z]*\n.*?\n```" # re.DOTALL

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# convert them to class
def get_json_list(file_path, start_at=0, end_at=None):
    with open(file_path, "r") as f:
        json_list = []
        for idx, line in enumerate(f):
            if end_at is not None and idx >= end_at:
                return json_list
            elif idx < start_at:
                continue
            json_list.append(json.loads(line))
        return json_list

def get_json_list_as_dict(file_path: str, key: Any):
    with open(file_path, "r") as f:
        json_dict = {}
        for line in f:
            tmp = json.loads(line)
            json_dict[tmp[key]] = tmp
        return json_dict
            

def update_json_dict(info_lst, idx, obj):
    for info_dict in info_lst:
        if info_dict['op_type'] == 'add':
            obj[info_dict['label']] = info_dict['lst'][idx]
        elif info_dict['op_type'] == 'del':
            del obj[info_dict['label']]
        elif info_dict['op_type'] == 'modify':
            obj[info_dict['label']] = obj[info_dict['old_label']]
            del obj[info_dict['old_label']]

def update_n_write_json_list(file_path, json_list, info_lst, io_type: str = "w+"):
    # a non-pandas naive way
    # file_path: out file path
    # json_list: input json_list
    # info_lst: [{"label": new_label, "lst": new_list, "op_type": "add"/"del"}, ...]
    # add: label, lst, op_type
    # del: label, op_type
    # modify: label, old_label, op_type
    with open(file_path, io_type) as out_f:
        for idx, obj in enumerate(json_list):
            update_json_dict(info_lst, idx, obj)
            json.dump(obj, out_f, ensure_ascii=False)
            out_f.write("\n")

def update_json_list(json_list, info_lst):
    for idx, obj in enumerate(json_list):
        update_json_dict(info_lst, idx, obj)

def store_json_list(file_path, json_list, is_obj=True):
    with open(file_path, 'w') as out_f:
        for obj in json_list:
            if is_obj:
                json.dump(obj, out_f, ensure_ascii=False)
            else:
                out_f.write(obj)
            out_f.write("\n")

def read_csv(path, if_empty_return=[]):
    if not os.path.isfile(path): return if_empty_return
    try:
        data = pd.read_csv(path).to_dict(orient="records")
        return data
    except:
        return if_empty_return

def write_dict_to_csv(data, file, mode='w'):
    if not len(data): return

    fieldnames = data[0].keys()
    lines = read_csv(file)
    existing = len(lines) >= 1
    if mode == 'a':
        if existing:
            fieldnames_existing = lines[0].keys()
            if set(fieldnames) - set(fieldnames_existing):
                print(f'[Warn] The existing csv columns ({fieldnames_existing}) are not compatible with the new csv '
                      f'columns ({fieldnames}).')
                import pdb;
                pdb.set_trace()
            fieldnames = fieldnames_existing
    with open(file, mode=mode) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if (mode == 'w') or (not existing): writer.writeheader()
        writer.writerows(data)

def ddict2dict(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = ddict2dict(v)
    return dict(d)

def get_url_root(url: str):
    url = url.split("//")[1]
    url = url.split("/")[0]
    url = url.replace("www.", "")
    return url

def get_file_str(file_path):
    with open(file_path, 'r') as f:
        return f.read()

def hash_url(url: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, url))

def set_seed(seed):
    import random
    import os

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

def _none_helper(func, *args, **kwargs):
    for arg in args:
        if arg is None:
            return
    func(*args, **kwargs)
    return

def _bool_helper(func, *args, **kwargs):
    for arg in args:
        if arg is False:
            return
    func(*args, **kwargs)
    return