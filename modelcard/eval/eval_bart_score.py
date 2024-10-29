import sys
import pandas as pd
import json
import evaluate
import os
from tqdm import tqdm

from modelcard.utils import get_json_list
from BARTScore.bart_score import BARTScorer

in_dir = sys.argv[1]
out_f = sys.argv[2]
print(in_dir, out_f)
files = os.listdir(in_dir)

out_df = []

scorer = BARTScorer(
    device="cuda:0",
    checkpoint="facebook/bart-large-cnn",
    cache_dir="YOUR_CACHE_DIR",
    max_length=1024
)
scorer.load(path="BARTScore/bart_score.pth")

if os.path.exists(out_f):
    exit()

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
        
        print(model_id, item['question'])
        questions = item['question']
        references = [item['prompt'][1]]
        predictions = [item['answer']]
        
        score = scorer.score(
            tgts=predictions,
            srcs=references,
            batch_size=1
        )

        row = {
            "model_id": model_id,
            "quesition": questions,
            "bart_score": score[0] 
        }
        
        out_df.append(row)

out_df = pd.DataFrame().from_records(out_df)
out_df.to_csv(out_f)