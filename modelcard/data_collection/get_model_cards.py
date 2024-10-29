# repocard_customized: cloned from huggingface-hub 0.17.3 repocard.py, adding the cache_dir support.

import huggingface_hub
import json
import argparse
import time
import os
from tqdm import tqdm
import sys
from modelcard.utils import get_json_list, update_json_list, store_json_list
from modelcard.data_collection.repocard_customized import ModelCard

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file",  type=str) # jsonl
    parser.add_argument("--out_dir",  type=str) # must ends with "/"
    # parser.add_argument("--type", type=str, choices=['model', 'dataset']) 
    parser.add_argument("--start_idx", type=int, default=0) 
    parser.add_argument("--end_idx", type=int, default=5000)
    parser.add_argument("--cache_dir", type=str, default="data/model_readme/")
    args = parser.parse_args() # exclusive end index
    return args

args = arg_parse()
model_list = get_json_list(
    args.in_file, 
    start_at=args.start_idx, 
    end_at=args.end_idx
)

key = 'modelId'
     
has_readme = []
for model in tqdm(model_list):
    id = model[key].replace("/", "_")
    assert id != ""
    try:
        card = ModelCard.load(model[key], cache_dir=args.cache_dir)
        with open(args.out_dir + id, 'w') as f:
            f.write(card.content)
        has_readme.append("classA")
    except huggingface_hub.utils._errors.EntryNotFoundError as e:
        print(e)
        has_readme.append("classC")
    except Exception as e:
        print(e)
        has_readme.append("classC")

update_json_list(
    model_list,
    [
        {
            "label": "class",
            "lst": has_readme,
            "op_type": "add",
        }
    ],
)

store_json_list(
    args.in_file.replace(".jsonl", "_full.jsonl"),
    model_list,
)