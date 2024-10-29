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
    model='gpt-3.5-turbo',
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
    # embedding_model_name='sentence-transformers/all-MiniLM-L6-v2',
    # embedding_model_name='sentence-transformers/all-mpnet-base-v2',
    embedding_model_name='jinaai/jina-embeddings-v2-base-en',
    database_name='faiss',
)

data = get_json_list('data/gen/overall/gpt_query_cache.jsonl')


import json
models = json.load(open("data/test_set.json", 'r'))
for model_id in models:
    model_id_file = model_id.replace("/", "_")

    args = load_model_args(model_id, args)
    gen = Generator(args, doc_args)
    
    out_f = open(f"data/gen/gpt_pipeline/{model_id_file}.jsonl", 'a')

    def write_f(info, out_f):
        json.dump(info, out_f)
        out_f.write("\n")
        out_f.flush()

    info = {
        "question": None,
        "chain": None,
        "prompt": None,
        "answer": None
    }

    for question in data:
        pseudo_answers, prompts = [], []
        question_config = getattr(gen.question_template, question['question'])
        if question_config['prompt'] == 'special':
            continue
        
        for sub_question in question['sub-questions']:
            pseudo_answer, prompt = gen.pseudo_answer_chain(sub_question)
            pseudo_answers.append(pseudo_answer)
            prompts.append(prompt)
        info['question'] = question['question']
        info['chain'] = 'pseudo_answer' # type: ignore
        info['prompt'] = prompts # type: ignore
        info['answer'] = pseudo_answers # type: ignore
        write_f(info, out_f)
        
        responses, prompts = gen.section_retrieval_chain(question['sub-questions'])
        info['question'] = question['question']
        info['chain'] = 'section_retrieval' # type: ignore
        info['prompt'] = prompts # type: ignore
        info['answer'] = responses # type: ignore
        write_f(info, out_f)
        
        filters = [{'heading': i} for i in responses]
        results, prompts = gen.retrieval_chain(pseudo_answers, k=5, filters=filters)
        info['question'] = question['question']
        info['chain'] = 'retrieval' # type: ignore
        info['prompt'] = prompts # type: ignore
        info['answer'] = results # type: ignore
        write_f(info, out_f)
        
        rev_filters = [{'[REVERSED]': True, 'heading': i} for i in responses]
        rev_results, rev_prompts = gen.retrieval_chain(pseudo_answers, k=5, filters=rev_filters)
        info['question'] = question['question']
        info['chain'] = 'rev_retrieval' # type: ignore
        info['prompt'] = rev_prompts # type: ignore
        info['answer'] = rev_results # type: ignore
        write_f(info, out_f)
        
        results_topn = results[:8] + rev_results[:3]
        results, prompts = gen.generation_chain(
            question_config, 
            "\n".join([f"{idx+1}. {i}" for idx, i in enumerate(results_topn)])
        )
        info['question'] = question['question']
        info['chain'] = 'generation' # type: ignore
        info['prompt'] = prompts # type: ignore
        info['answer'] = results # type: ignore
        write_f(info, out_f)
    
    
    # convert records to model card
    records = get_json_list(f"data/gen/gpt_pipeline/{model_id_file}.jsonl")
    res_str = ""
    for question in gen.question_template.list_variables():
        question_config = getattr(gen.question_template, question)
        if question_config['prompt'] == 'special':
            if question == 'developed_by':
                res = gen.doc_store.get_section("authors")
                res = res if res is not None else "[More Information Needed]"
                res_str += question_config['markdown_header'].format(answer=res)
            elif question == 'model_sources':
                paper_link = gen.paper_link if gen.paper_link is not None else "[More Information Needed]"
                github_link = gen.github_link if gen.github_link is not None else "[More Information Needed]"
                res_str += question_config['markdown_header'].format(paper_link=paper_link, github_link=github_link)
            elif question == 'citation_bibtex': 
                title = gen.doc_store.get_section("title")
                authors = gen.doc_store.get_section("authors")
                url = gen.paper_link
                res = question_config['helper_func'](title, authors, url)
                res_str += question_config['markdown_header'].format(answer=res)
            continue
        for record in records:
            if record['chain'] == 'generation' and record['question'] == question:
                res_str += question_config['markdown_header'].format(answer=record['answer'], model=gen.model_id)
                break
        else:
            # print(question)
            raise NotImplementedError
            # continue
    
    # store the converted model card
    out_f = open(f"data/generated_model_cards/gpt3.5/{model_id_file}.md", 'w')
    out_f.write(res_str)
    out_f.flush()