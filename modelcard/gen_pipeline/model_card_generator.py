# Input: Prompt template, generation config arguments, which text model to use, cache
# Output: Generated text

import argparse
import json
import openai
import os
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import re
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from modelcard.gen_pipeline.prompt_template import Template, QuestionTemplate
from modelcard.gen_pipeline.paragraph_store import ParagraphStore
from modelcard.utils import get_json_list_as_dict, set_seed

def arg_parse():
    parser = argparse.ArgumentParser()
    
    # model port
    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument("--api_base", type=str, default="http://localhost:8000/v1")
    
    # model id
    parser.add_argument("--model_id", type=str)
    
    # paper doc
    parser.add_argument("--paper_path", type=str, default=None, help='the paper path')
    parser.add_argument("--paper_type", type=str, choices=['markdown', 'txt', 'json'], default='json', help='the paper type')
    
    # github doc
    parser.add_argument("--github_path", type=str, default=None, help='the github readme path')
    parser.add_argument("--github_type", type=str, choices=['markdown', 'txt', 'json'], default='markdown', help='the github readme type')
    
    # generated file storing path
    parser.add_argument("--output_path", type=str, help='the output model card path')
    
    # model configs
    parser.add_argument("--model", choices=['gpt-3.5-turbo-0613', 'llama2-70b-chat'], type=str, default='llama2-70b-chat', help='use which model to generate model cards')
    parser.add_argument("--self_host", action='store_true', default=True)
    parser.add_argument("--chat_mode", action='store_true', default=False)
    
    # generation configs
    parser.add_argument("--section", type=str, default='all', help='which section to generate. By default it generate all sections') 
    args = parser.parse_args()

    return args

def load_model_args(model_id: str | None, args):
    if model_id is not None:
        args.model_id = model_id
        args.paper_type = 'json'
        card2pdf = get_json_list_as_dict("data/model_card2pdf.jsonl", key='modelId')
        tmp = card2pdf.get(model_id, {'direct_paper_hash': None})['direct_paper_hash']
        if tmp is None:
            args.paper_path = None
        else:
            args.paper_path = "data/total/scipdf/" + Path(tmp).stem + ".json"
        
        card2github = get_json_list_as_dict("data/model_card2github_readme.jsonl", key='modelId')
        tmp = card2github.get(model_id, {'direct_github_hash': None})['direct_github_hash']
        if tmp is None:
            args.github_path = None
        else:
            args.github_path = "data/total/github/" + Path(tmp).stem + ".md"
        
        args.output_path = "data/generated_model_cards/" + model_id.replace("/", "_") + '.md'
        return args
    else:
        args.model_id = None
        args.paper_type = 'json'
        args.paper_path = None
        args.github_path = None
        return args

class ContentSampler():
    """Used for few-shot & vector retrieval"""
    def __init__(self):
        self.load_pool()
    
    def load_pool(self):
        self.pool = []
        for filename in os.listdir(self.pool_dir): # type: ignore
            path = os.path.join(self.pool_dir, filename) # type: ignore
            self.pool.append(json.load(path))
    
    def sample(self, n, question_type, seed=0):
        set_seed(seed)
        question_pool = []
        for model in self.pool:
            ans = model.get(question_type, None)
            if ans is not None:
                question_pool.append(ans)
        
        sample_n = min(len(question_pool), n)
        print(f">>>Sampler samples with size {sample_n}..")
        return np.random.choice(question_pool, sample_n, replace=False).tolist()
        

class Generator():
    def __init__(self, args, doc_args):
        self.args = args
        self.doc_args = doc_args
        # initialize llm models in openai mode
        if args.use_llm:
            if args.model.startswith("gpt"):
                # goes to openai
                # openai.api_key = os.getenv("OPENAI_API_KEY")
                openai.api_key = args.api_key
                openai.organization = args.organization
                llm_model = None
                if args.chat_mode:
                    from langchain.chat_models import ChatOpenAI
                    llm_model = ChatOpenAI
                else:
                    from langchain.llms.openai import OpenAI
                    llm_model = OpenAI

                self.llm = llm_model(
                    openai_api_key=args.api_key,
                    model=args.model,
                ) # type: ignore
            elif args.model.startswith("claude"):
                llm_model = None
                if args.chat_mode:
                    from langchain_anthropic import ChatAnthropic
                    llm_model = ChatAnthropic
                else:
                    raise NotImplementedError
                
                self.llm = llm_model(
                    anthropic_api_key=args.api_key, # type: ignore
                    model_name=args.model
                )
            else:
                if args.self_host:
                    # goes to self-hosted models
                    openai.api_key = args.api_key
                    openai.api_base = args.api_base
                    llm_model = None
                    if args.chat_mode:
                        from modelcard.gen_pipeline.chat_openai import ChatVLLMOpenAI
                        llm_model = ChatVLLMOpenAI
                    else:
                        from langchain.llms.vllm import VLLMOpenAI
                        llm_model = VLLMOpenAI
                    
                    if args.vllm:
                        self.llm = llm_model(
                            openai_api_key=args.api_key,
                            openai_api_base=args.api_base,
                            model_name=openai.Model.list()['data'][0]['id'], # type: ignore
                            # model_kwargs={"stop": ["."]},
                            max_tokens = 512,
                        )
                    else:
                        self.llm = llm_model(
                            openai_api_key=args.api_key,
                            openai_api_base=args.api_base,
                            # model_kwargs={"stop": ["."]},
                            max_tokens = 512,
                        )
                else:
                    raise NotImplementedError("Not implemented")
            self.llm.temperature = args.temperature
        
        
        self.generation_args = {}
        self.template = Template()
        self.question_template = QuestionTemplate()
        

    def load_doc(self):
        # initialize doc_store object
        self.doc_store = ParagraphStore(
            self.args,
            self.doc_args,
        )
        self.doc_store.load_doc()
        
        self.get_link()
        
    def get_link(self):
        self.model_id = self.args.model_id
        
        paper_link_dict = get_json_list_as_dict('data/als/link_auto.jsonl', 'modelId')
        self.paper_link = paper_link_dict.get(self.model_id, {}).get('paper_urls', [])
        self.paper_link = self.paper_link[0] if len(self.paper_link) > 0 else None
        
        github_link_dict = get_json_list_as_dict('data/als/link_auto_github.jsonl', 'modelId')
        self.github_link = github_link_dict.get(self.model_id, {}).get('github_urls', [])
        self.github_link = self.github_link[0] if len(self.github_link) > 0 else None
    

    def print_prompt(self, prompt):
        print(">>>Prompt:")
        [print(i.content) for i in prompt.messages]
        return [i.content for i in prompt.messages]
    
    def print_response(self, response):
        print(">>>Response:")
        print(response)
    
    def extract_and_eval_list(self, input_string):
        """
        Extracts a substring in Python list format from the original string and evaluates it to return the Python list.
        If it cannot be extracted or if the extraction leads to an error, it returns False.
        """
        try:
            start = input_string.find('[')
            end = input_string.rfind(']') + 1
            if start == -1 or end == 0:
                return False
            list_str = input_string[start:end]
            extracted_list = eval(list_str)
            if isinstance(extracted_list, list):
                return extracted_list
            else:
                return False
        except:
            return False
    
    def extract_markdown_list(self, input_string):
        """
        Extracts a list of bullet points in markdown format from the original string and converts it to a Python list.
        If it cannot be extracted or if any error occurs, it returns False.
        """
        try:
            # Regular expression pattern to match markdown list items
            pattern = r'-\s*(.+?)\s*(?:\n|$)'

            # Find all matches using the pattern
            matches = re.findall(pattern, input_string)

            # If no matches found, return False
            if not matches:
                return False

            return matches
        except Exception as e:
            # Return False in case of any error
            return False
        
    def process_response(self, query, resp, type='python_list'):
        """
        the response is in python list format or in markdown bullet point format
        
        - type: "python_list" or "bullet_point", else None
        """
        resp = resp.strip()
        query_list = []
        
        if type == 'python_list':
            resp = self.extract_and_eval_list(resp)
        elif type == 'bullet_point':
            resp = self.extract_markdown_list(resp)
        else:
            raise NotImplementedError("Other answer types are not implemented for postprocessing!")
            
        if resp == False:
            query_list = query
        else:
            query_list = resp
        return query_list
    
    def division_chain(self, question_config):
        """
        divide the query into sub-queries
        """
        query = question_config['prompt'].format(model="")
        prompt = self.template.get_division_prompt(query)
        chat_template = ChatPromptTemplate.from_messages(prompt)
        chain = (chat_template | self.llm)
        response = chain.invoke({})
        response = self.process_response(
            [query], 
            response.content, 
            type='python_list'
        )
        return response
        
    def pseudo_answer_chain(self, query, model=True):
        """
        generate pseudo answer for each sub-query
        """
        model_id = self.model_id if model else None
        prompt = self.template.get_pseudo_answer_prompt(query, model=model_id)
        chat_template = ChatPromptTemplate.from_messages(prompt)
        prompts = self.print_prompt(chat_template)
        chain = (chat_template | self.llm)
        response = chain.invoke({})
        response = response.content
        return response, prompts
    
    def retrieval_chain(self, pseudo_answers, k=10, fetch_k=1000, filters=None):
        """
        retrieve relevant documents for each "pseudo answer" query
        """
        
        results = {}
        if filters is None:
            filters = [None] * len(pseudo_answers)
        for pseudo_answer, filter in zip(pseudo_answers, filters):
            docs = self.doc_store.search(
                pseudo_answer,
                k=k,
                filter=filter,
                fetch_k=fetch_k,
            )
            for doc in docs: # type: ignore
                key = doc[0].page_content          
                if (key in results and results[key][0] > doc[1]) or (key not in results):
                    results[key] = (doc[1], doc[0].metadata)
            
        print(results)
        return list(results.keys()), pseudo_answers
    
    def section_retrieval_chain(self, sub_queries):
        """
        Optional
        retrieve relevant sections from documents first for each sub query
        Then, please call retrieval_chain() to retrieve relevant chunks from these sections with each corresponding pseudo answer
        """
        def sanity_check(response: list, headers):
            """
            check whether each section header in the list can be found from orginal papers. Otherwise omit that section header.
            """
            return [item for item in response if item in headers]
        
        model_id = self.model_id
        paper_headers = self.doc_store.get_headers(source='paper')
        github_headers = self.doc_store.get_headers(source='github')
        
        all_responses = []
        all_prompts = []
        for sub_query in sub_queries:
            prompt = self.template.get_section_retrieval_prompt(
                sub_query,
                model_id,
                paper_headers if paper_headers != [] else None,
                github_headers if github_headers != [] else None,
            )
            chat_template = ChatPromptTemplate.from_messages(prompt)
            prompts = self.print_prompt(chat_template)
            all_prompts.append(prompts)
            chain = (chat_template | self.llm)
            response = chain.invoke({})
            response = self.process_response(
                [], 
                response.content, 
                type='bullet_point'
            )
            response = sanity_check(response, paper_headers + github_headers)
            if len(response) > 5:
                response = []
            print(response)
            all_responses.append(response)
        if [i for j in all_responses for i in j] == []:
            all_responses = [paper_headers + github_headers]
        return all_responses, all_prompts
    
    def section_retrieval_with_keyword_chain(self, sub_queries, question_config):
        """
        Optional
        retrieve relevant sections from documents first for each sub query
        Then, please call retrieval_chain() to retrieve relevant chunks from these sections with each corresponding pseudo answer
        """
        def sanity_check(response: list, headers):
            """
            check whether each section header in the list can be found from orginal papers. Otherwise omit that section header.
            """
            return [item for item in response if item in headers]
        
        model_id = self.model_id
        paper_headers = self.doc_store.get_headers(source='paper')
        github_headers = self.doc_store.get_headers(source='github')
        
        all_responses = []
        all_prompts = []
        for sub_query in sub_queries:
            prompt = self.template.get_section_retrieval_prompt(
                sub_query,
                model_id,
                paper_headers if paper_headers != [] else None,
                github_headers if github_headers != [] else None,
            )
            chat_template = ChatPromptTemplate.from_messages(prompt)
            prompts = self.print_prompt(chat_template)
            all_prompts.append(prompts)
            chain = (chat_template | self.llm)
            response = chain.invoke({})
            response = self.process_response(
                [], 
                response.content, 
                type='bullet_point'
            )
            response = sanity_check(response, paper_headers + github_headers)
            if len(response) > 5:
                response = []
            # add keyword section extraction
            for keyword in question_config['keyword']:
                for header in paper_headers + github_headers:
                    if keyword in header:
                        response.append(header)
            response = list(set(response))
            
            print(response)
            all_responses.append(response)
        if [i for j in all_responses for i in j] == []:
            all_responses = [paper_headers + github_headers]
        return all_responses, all_prompts
    
    def generation_chain(self, question_config, ref_str, model=True):
        model_str = f" {self.model_id}" if model else ""
        prompt = self.template.get_generation_prompt(
            question_config['prompt'].format(model=model_str),
            self.model_id,
            ref_str,
            question_config['role']
        )
        chat_template = ChatPromptTemplate.from_messages(prompt)
        prompts = self.print_prompt(chat_template)
        chain = (chat_template | self.llm)
        response = chain.invoke({})
        response = response.content
        return response, prompts

    def direct_generation_chain(self, prompt):
        assert len(prompt) == 2
        chat_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=prompt[0]),
            HumanMessage(content=prompt[1])
        ])
        prompts = self.print_prompt(chat_template)
        assert prompt == prompts
        chain = (chat_template | self.llm)
        response = chain.invoke({})
        response = response.content
        return response, prompts
    
    def markdown_template_chain(self, question_config, response):
        
        response = question_config['markdown_header'].format(answer=response, model=self.model_id)
        return response