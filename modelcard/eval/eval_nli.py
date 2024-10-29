import pandas as pd
import json
import sys
import evaluate
import os
from tqdm import tqdm
from sentence_transformers import CrossEncoder

from modelcard.utils import get_json_list

model = CrossEncoder('cross-encoder/nli-deberta-v3-large', automodel_args={'cache_dir': 'YOUR_CACHE_DIR'})
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
        # assert len(data) == 31

        for item in tqdm(data):
            if item['chain'] != 'generation':
                continue
            
            questions = item['question']
            references = item['prompt'][1]
            predictions = item['answer']
            
            score = model.predict([
                (references,
                predictions)
            ])
            label_mapping = ['contradiction', 'entailment', 'neutral']
            labels = [label_mapping[score_max] for score_max in score.argmax(axis=1)]

            row = {
                "model_id": model_id,
                "quesition": questions,
                "nli_pred": labels[0] 
            }
            
            out_df.append(row)

    out_df = pd.DataFrame().from_records(out_df)
    out_df.to_csv(out_f)
