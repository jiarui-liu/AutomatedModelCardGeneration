# Get paper link from hugging face readme;
# Get github link from hugging face readme.

import abc
import typing
from typing import Union
import sys
import json
import re
import numpy as np
from collections import Counter
import bibtexparser
from functools import reduce
from modelcard.utils import md_code_block_regex, url_regex, github_url_regex, arxiv_id_regex, get_url_root

class ModelCardParser():
    """ModelCardParser
    """
    def __init__(self, content: str):
        self.content=content
    
    @staticmethod
    def remove_md_code_block_descriptor(block):
        block = block.strip()
        re.sub(r"```[^\S\r\n]*[a-z]*\n", "", block)
        re.sub(r"[\r\n ]*```", "", block)
        return block
    
    def get_citation_str(self) -> str:
        code_blocks = re.finditer(md_code_block_regex, self.content, flags=re.DOTALL)
        idx_list = []
        tmp = []
        for block in code_blocks:
            if len(re.findall(r"@\w*{.*[, }]", block.group(0))) > 0:
                tmp.append(self.remove_md_code_block_descriptor(block.group(0)))
                idx_list.append([block.start(), block.end()])
        code_blocks = "\n\n".join(tmp)
        # a long string, the start index of the first bibtex block
        return code_blocks, idx_list # type: ignore
    
    def parse_bib(self, citation_str : str = None) -> bibtexparser.Library: # type: ignore
        """Return a parsed bib list

        Args:
            citation_str (str, optional): Defaults to None.

        Returns:
            bibtexparser.Library
        """
        if citation_str is None:
            citation_str, _ = self.get_citation_str()
            if citation_str == "": return
        
        lib = bibtexparser.parse_string(citation_str)
        # only use url or title for searching
        
        return lib 


def format_arxiv_url(arxiv_tag: str = None) -> str: # type: ignore
    arxiv_url = 'https://arxiv.org/abs/' + arxiv_tag.split(":")[-1]
    return arxiv_url

def format_arxiv_pdf_url(arxiv_url: str) -> str:
    if is_arxiv_link(arxiv_url):
        arxiv_url = arxiv_url.replace("/abs/", "/pdf/") + ".pdf"
    return arxiv_url

def format_hf_model_card_url(model_id: str) -> str:
    url = f"https://huggingface.co/{model_id}"
    return url

def is_arxiv_link(url: str) -> bool:
    if 'arxiv.org' in url or 'arxiv:' in url:
        return True
    return False

def format_paper_pdf_url(url: str) -> Union[str, None]:
    if is_arxiv_link(url):
        arxiv_ids = re.findall(arxiv_id_regex, url)
        print(arxiv_ids, url)
        if len(arxiv_ids) == 0:
            return None
        assert len(arxiv_ids) == 1
        res = format_arxiv_pdf_url(format_arxiv_url(arxiv_ids[0]))
        return res
    # check paper pdf types one by one
    url = url.strip("/")
    if "anthology.org" in url:
        if url.endswith(".pdf"):
            return url
        return url + ".pdf"
    if "openaccess.thecvf.com" in url:
        url = url.replace("/html/", "/papers/")
        url = url.replace(".html", ".pdf")
        return url
    return None

# get paper link from tags and from hugging face readme
class LinkGetter():
    def __init__(self, paper_url_set_path):
        self.paper_url_set_path = paper_url_set_path
    
    @staticmethod
    def get_arxiv_id_from_tags(tags: list[str]) -> list[str]:
        arxiv_ids = []
        for tag in tags:
            if is_arxiv_link(tag):
                arxiv_ids.append(tag)
        return arxiv_ids
    
    @staticmethod
    def get_paper_link_from_tags(tags: list[str]) -> list[str]:
        arxiv_ids = LinkGetter.get_arxiv_id_from_tags(tags)
        paper_links = []
        for tag in arxiv_ids:
            paper_links.append(format_arxiv_url(arxiv_tag=tag))
        return paper_links
    
    def get_paper_urls_list(self, exclude=['huggingface.co', 'github.com']):
        urls_list = json.load(open(self.paper_url_set_path, 'r'))
        for url in exclude:
            if url in urls_list:
                urls_list.remove(url)
        return urls_list
    
    def get_paper_link_from_hf_page(self, tags: list[str], content: str) -> tuple[list]:
        if content == "":
            return ([], [], []) # type: ignore
        original_urls = []
        # subtasks:
        # 1. arxiv_ids
        arxiv_ids = LinkGetter.get_arxiv_id_from_tags(tags)
        
        # 2. other urls than huggingface
        parser = ModelCardParser(content)
        code_blocks, idx_list = parser.get_citation_str()
        other_urls = re.findall(url_regex, content)
        original_urls.extend(other_urls)
        
        other_paper_urls = self.get_paper_urls_list()
        other_urls = [url for url in other_urls if get_url_root(url) in other_paper_urls]
        original_urls = [url for url in original_urls if get_url_root(url) in other_paper_urls]
        for arxiv_id in arxiv_ids:
            flag = False
            for url in other_urls:
                if arxiv_id in url:
                    flag = True
            if not flag:
                other_urls.append(format_arxiv_url(arxiv_id))
                original_urls.append(arxiv_id)
        print("other_urls:", other_urls)
        
        # 3. citation arxiv_ids if it doesn't have urls being extracted before
        lib = parser.parse_bib(citation_str=code_blocks)
        if lib is not None:
            for k, v in lib.entries_dict.items():
                flag = False
                for field in v.fields:
                    res = list(set(re.findall(arxiv_id_regex, field._value)))
                    if len(res) > 0:
                        assert len(res) == 1
                        if res[0] not in arxiv_ids:
                            other_urls.append(format_arxiv_url(res[0]))
                            original_urls.append(res[0])
                            flag = True
                        break
                if flag:
                    break
        
        # aggregate
        # return a list of final indices, and formatted_paper_urls
        return self.aggregate(other_urls, content, lib, original_urls)
    
    def aggregate(self, other_urls_prev: list, content: str, lib: bibtexparser.Library, original_urls) -> tuple[list]:
        if len(other_urls_prev) == 0:
            return ([], [], []) # type: ignore

        print("other_urls_prev:", other_urls_prev)
        formatted_paper_urls = []
        formatted_original_urls = []
        other_urls = []
        for url, original_url in zip(other_urls_prev, original_urls):
            formatted_url = format_paper_pdf_url(url)
            if formatted_url is not None:
                formatted_paper_urls.append(formatted_url)
                formatted_original_urls.append(original_url)
                other_urls.append(url)

        if len(formatted_paper_urls) == 1:
            return [0], formatted_paper_urls, formatted_original_urls # type: ignore
        else:
            paper_ranks = {url: [1, idx] for idx, url in enumerate(formatted_paper_urls)}
            print(paper_ranks)
            # step 1: count the number of occurrences 1, 2, 3, ...
            counter = Counter(formatted_paper_urls)
            for k in counter:
                paper_ranks[k][0] *= counter[k] * 2
            print(paper_ranks)
            # step 2: the place of occurrence 3: beginning, 2: end, 1: elsewhere
            # url occurrences
            url_first_occurrences = {}
            for idx, url in enumerate(other_urls):
                # https://www.geeksforgeeks.org/python-all-occurrences-of-substring-in-string/
                if is_arxiv_link(url):
                    url = re.findall(arxiv_id_regex, url)[0]
                occurrences = re.finditer(url, content)
                res = reduce(lambda x, y: x + [y.start()], occurrences, [])
                print(res, url)
                assert len(res) >= 1
                url_first_occurrences[formatted_paper_urls[idx]] = res[0]
            
            url_first_occurrences = {k: v for k, v in sorted(url_first_occurrences.items(), key=lambda item: item[1])}
            for idx, url in enumerate(url_first_occurrences):
                paper_ranks[url][0] += len(url_first_occurrences) - idx
            
            # url occurrences in bibtex
            if lib is not None:
                url_first_occurrences_in_lib = {}
                for idx, (k, v) in enumerate(lib.entries_dict.items()):
                    for field in v.fields:
                        tmp = None
                        for idx_j, url in enumerate(other_urls):
                            if url in field._value and formatted_paper_urls[idx_j] not in url_first_occurrences_in_lib:
                                url_first_occurrences_in_lib[formatted_paper_urls[idx_j]] = idx
                                tmp = formatted_paper_urls[idx_j]
                                break
                        if tmp in url_first_occurrences_in_lib:
                            break
                url_first_occurrences_in_lib = {k: v for k, v in sorted(url_first_occurrences_in_lib.items(), key=lambda item: item[1])}
                for idx, url in enumerate(url_first_occurrences_in_lib):
                    paper_ranks[url][0] += len(url_first_occurrences_in_lib) - idx
            
            # [TODO] step 3: semantic classifier
            # a reversely sort the data structure 
            paper_ranks = {k: v for k, v in sorted(paper_ranks.items(), key=lambda item: item[1][0], reverse=True)}
            print(paper_ranks)
            top_v = None
            k_s = []
            for idx, (k, v) in enumerate(paper_ranks.items()):
                if idx == 0:
                    top_v = v[0]
                    k_s.append(v[1])
                elif v[0] == top_v:
                    k_s.append(v[1])
            return (k_s, formatted_paper_urls, formatted_original_urls) # type: ignore
        
    
    def get_paper_link(self, content_hf: Union[str, None], content_github: Union[str, None], tags: Union[list[str], None]) -> Union[str, None]:
        lst = [
            self.get_paper_link_from_tags(tags), # type: ignore
            self.get_paper_link_from_hf_page(content_hf), # type: ignore
            self.get_paper_link_from_github_page(content_github), # type: ignore
        ]
        return self.aggregate(lst) # type: ignore
    

class GithubGetter():
    def __init__(self):
        pass
    
    def get_github_link_from_hf_page(self, content: str):
        github_links = re.findall(github_url_regex, content)
        github_links = {
            item: github_links.count(item) for item in set(github_links)
        }
        github_links = {k: v for k, v in sorted(github_links.items(), key=lambda item: item[1], reverse=True)}
        best_links = []
        other_links = []
        max_num_occurrences = 0
        for idx, link in enumerate(github_links):
            if idx == 0:
                best_links.append(link)
                max_num_occurrences = github_links[link]
            else:
                if github_links[link] == max_num_occurrences:
                    best_links.append(link)
                else:
                    other_links.append(link)
        return best_links, other_links


def get_paper_link_from_file(content: str, res=None):
    pass
