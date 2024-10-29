# https://github.com/lukasschwab/arxiv.py
# https://medium.com/@tiagotoledojr/creating-an-arxiv-crawler-with-python-and-scrapy-dedbe79e7ce6

import logging, arxiv
import requests
import json
import time
from pathlib import Path
import os

class Spider():
    def __init__(self, error_file):
        self.error_file = error_file
        self.f = open(self.error_file, 'a')
    
    def log_error(self, error_info):
        json.dump(error_info, self.f, ensure_ascii=False)
        self.f.write("\n")
        

class ArxivSpider(Spider):
    def __init__(self, error_file, max_tries=2):
        super().__init__(error_file)
        self.max_tries = max_tries
    
    def download_pdf(self, arxiv_id: str, dir_path, file_hash):
        curr_tries = 0
        while curr_tries < self.max_tries:
            try:
                paper = next(arxiv.Search(id_list=[arxiv_id]).results())
                paper.download_pdf(dirpath=dir_path, filename=str(Path(dir_path) / f"{file_hash}.pdf"))
                break
            except Exception as e:
                print(e)
            curr_tries += 1
            time.sleep(5)
        if curr_tries == self.max_tries:
            self.log_error({
                "type": "arxiv_spider",
                "arxiv_id": arxiv_id,
                "dir_path": dir_path,
                "file_hash": file_hash,
            })

class PdfSpider(Spider):
    def __init__(self, error_file, max_tries=3):
        super().__init__(error_file)
        self.max_tries = max_tries
    
    def download_pdf(self, url, dir_path, file_hash):
        curr_tries = 0
        while curr_tries < self.max_tries:
            try: 
                r = requests.get(url, timeout=1, verify=True) 
                r.raise_for_status()
                with open(str(Path(dir_path) / f"{file_hash}.pdf"), 'wb') as f:
                    f.write(r.content)
                break
            except requests.exceptions.HTTPError as errh: 
                print("HTTP Error", errh) 
                print(errh.args[0]) 
            except requests.exceptions.ReadTimeout as errrt: 
                print("Time out", errrt) 
            except requests.exceptions.ConnectionError as conerr: 
                print("Connection error", conerr) 
            except requests.exceptions.RequestException as errex: 
                print("Exception request", errex) 
            except Exception as e:
                print(e)
            curr_tries += 1
            time.sleep(5)
        if curr_tries == self.max_tries:
            self.log_error({
                "type": "pdf_spider",
                "url": url,
                "dir_path": dir_path,
                "file_hash": file_hash,
            })

class GitHubReadmeSpider(Spider):
    def __init__(self, error_file, max_tries=3):
        super().__init__(error_file)
        self.max_tries = max_tries
    
    def get_readme(self, combos):
        token = os.environ['GITHUB_TOKEN']
        owner = combos[0]
        repo = combos[1]
        dir = combos[2]

        url = f"https://api.github.com/repos/{owner}/{repo}/readme/{dir}"

        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28"
        }

        curr_tries = 0
        result = None
        while curr_tries < self.max_tries:
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                result = eval(response.text)['_links']['html']
                break
            else:
                curr_tries += 1
                print(f"Request failed with status code {response.status_code}")
                time.sleep(5)
        
        if curr_tries == self.max_tries and result is None:
            self.log_error({
                "type": "github_getter",
                "url": url,
                "combos": combos
            })
        return result
    
    def download_pdf(self, link, url, dir_path, file_hash):
        curr_tries = 0
        while curr_tries < self.max_tries:
            try: 
                r = requests.get(url, timeout=1, verify=True) 
                r.raise_for_status()
                with open(str(Path(dir_path) / f"{file_hash}.md"), 'wb') as f:
                    f.write(r.content)
                break
            except requests.exceptions.HTTPError as errh: 
                print("HTTP Error", errh) 
                print(errh.args[0]) 
            except requests.exceptions.ReadTimeout as errrt: 
                print("Time out", errrt) 
            except requests.exceptions.ConnectionError as conerr: 
                print("Connection error", conerr) 
            except requests.exceptions.RequestException as errex: 
                print("Exception request", errex) 
            except Exception as e:
                print(e)
            curr_tries += 1
            time.sleep(5)
        if curr_tries == self.max_tries:
            self.log_error({
                "type": "github_spider",
                "url": url,
                "dir_path": dir_path,
                "file_hash": file_hash,
            })