from types import SimpleNamespace
import openai
import time
import sys
from modelcard.utils import get_json_list_as_dict, get_json_list
from modelcard.gen_pipeline.model_card_generator import * # type: ignore

args = SimpleNamespace(
    api_key="YOUR_API_KEY",
    organization="YOUR_ORGANIZATION_ID",
    github_path=None, 
    github_type='markdown', 
    model='gpt-4-0613',
    section='all',
    paper_type='json',
    output_path='',
    temperature=0,
    chat_mode=True,
)
doc_args = SimpleNamespace(
    splitter='recursive',
    chunk_size=512,
    chunk_overlap=0,
    embedding_model_name='jinaai/jina-embeddings-v2-base-en',
    database_name='faiss',
)

args = load_model_args("LOAD_AN_EXAMPLE_MODEL_CARD_BY_ID", args)
gen = Generator(args, doc_args)

f = 'data/gen/overall/gpt_query_cache.jsonl'

import json
f = open(f, 'a')
questions = gen.question_template.list_variables()
mapping = gen.question_template.get_mapping()
for question in questions:
    attr = mapping[question]
    if attr['prompt'] == 'special':
        continue
    print(question)
    resp = gen.division_chain(attr)
    print(resp)
    json.dump({"question": question, "sub-questions": resp}, f)
    f.write("\n")
    f.flush()