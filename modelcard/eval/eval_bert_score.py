import pandas as pd
import json
import sys
import evaluate
import os
from tqdm import tqdm

from modelcard.utils import get_json_list
from bert_score import score

in_dir = sys.argv[1]
out_f = sys.argv[2]
print(in_dir, out_f)
if os.path.exists(out_f):
    exit()

files = os.listdir(in_dir)

out_df = []

for f in tqdm(files):
    if f.startswith("."):
        continue
    model_id = f.replace(".jsonl", "").replace("_", "/")
    f_path = os.path.join(in_dir, f)
    data = get_json_list(f_path)
    # assert len(data) == 31
    

    for item in tqdm(data):
        if item['chain'] != 'generation':
            continue
        
        questions = item['question']
        references = [item['prompt'][1]]
        predictions = [item['answer']]
        
        bert_p, bert_r, bert_f1 = score(
            predictions,
            references,
            model_type="microsoft/deberta-v3-base",
            lang='en',
            verbose=True,
            num_layers=9,
        )

        row = {
            "model_id": model_id,
            "quesition": questions,
            "bert_score_P": bert_p.item(),
            "bert_score_R": bert_r.item(),
            "bert_score_F1": bert_f1.item(),     
        }
        
        out_df.append(row)

out_df = pd.DataFrame().from_records(out_df)
out_df.to_csv(out_f)