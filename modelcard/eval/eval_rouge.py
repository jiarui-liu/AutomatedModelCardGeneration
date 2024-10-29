import pandas as pd
import json
import sys
import evaluate
import os
from tqdm import tqdm

from modelcard.utils import get_json_list

rouge = evaluate.load('rouge')

in_dir = sys.argv[1]
out_f = sys.argv[2]

def batch_process(in_dir, out_f):
    print(in_dir, out_f)
    
    files = os.listdir(in_dir)
    out_df = []
    
    if os.path.exists(out_f):
        return
    
    for f in tqdm(files):
        if f.startswith("."):
            continue
        model_id = f.replace(".jsonl", "").replace("_", "/")
        f_path = os.path.join(in_dir, f)
        data = get_json_list(f_path)
        for item in tqdm(data):
            if item['chain'] != 'generation':
                continue
            
            references = [item['prompt'][1]]
            predictions = [item['answer']]
            rouge_res = rouge.compute(
                predictions=predictions,
                references=references
            )
            row = {
                "model_id": model_id,
                "quesition": item['question'],
                "rouge1": rouge_res['rouge1'],
                "rouge2": rouge_res['rouge2'],
                "rougeL": rouge_res['rougeL'],
                "rougeLsum": rouge_res['rougeLsum']        
            }
            out_df.append(row)

    out_df = pd.DataFrame().from_records(out_df)
    out_df.to_csv(out_f)
