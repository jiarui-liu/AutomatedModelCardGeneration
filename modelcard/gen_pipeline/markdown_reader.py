import re
import markdown
from bs4 import BeautifulSoup

class DocTree():
    def __init__(self):
        self.index = None
        self.header = None
        self.parent = None
        self.children = []
        
    def insert_child(self, idx, header):
        new_node = DocTree()
        new_node.parent = self
        new_node.index = idx
        new_node.header = header
        self.children.append(new_node)
        return new_node

class MarkdownReader():
    def __init__(self, doc_path=None, doc=None, repl_code=False):
        assert doc_path is not None or doc is not None
        self.repl_code = repl_code
        if doc_path is not None: 
            self.doc_path = doc_path
            with open(doc_path, 'r') as file:
                self.doc = file.read()
        else:
            self.doc = doc
        self.to_html()
        self.get_headers()
    
    def to_html(self):
        self.doc_html = markdown.markdown(self.doc) # type: ignore
        self.soup = BeautifulSoup(self.doc_html, 'html.parser')
    
    def get_code_blocks(self, section):
        # Regular expression pattern to find markdown code blocks
        pattern = r'```(.*?)```'
        repl_str = '```\n[code block]\n```'
        # Finding all matches
        matches = re.finditer(pattern, section, re.DOTALL)

        # Constructing the dictionary to store code blocks
        code_blocks = {}
        for idx, match in enumerate(matches):
            if idx == 0:
                start_index = match.start()
            else:
                length = len(code_blocks)
                start_index = match.start() - sum([len(i) for i in code_blocks.values()]) + len(repl_str) * length
            code_block = match.group()
            code_blocks[start_index] = code_block

        # Substituting original code blocks with "[code block]"
        substituted_string = re.sub(pattern, repl_str, section, flags=re.DOTALL)

        return substituted_string, code_blocks
    
    def get_headers(self):
        # naive approach: regex
        header_levels = ["#", "##", "###", "####", "#####", "######"]
        res = [i for i in re.finditer(r'(^(#{1,6})\s+(.*))', self.doc, re.MULTILINE)] # type: ignore
        self.sections = []
        # chunk the doc
        for idx, i in enumerate(res):
            section = self.doc[i.end():res[idx+1].start()] if idx < len(res) - 1 else self.doc[i.end():] # type: ignore
            if self.repl_code:
                section, code_blocks = self.get_code_blocks(section)
            
                section_info = {
                    "heading": i.groups()[0].strip(),
                    "content": section,
                    "source": 'github',
                    "header_level": i.groups()[1].strip(),
                    "code_blocks": code_blocks
                }
                
            else:
                section_info = {
                    "heading": i.groups()[0].strip(),
                    "content": section,
                    "source": 'github',
                    "header_level": i.groups()[1].strip(),
                }
                
            self.sections.append(section_info)
    
    def find_parent(self, section, section_index):
        idx = section_index - 1
        while idx >= 0 and self.sections[idx]['header_level'] >= section['header_level']:
            idx -= 1
        
        if idx < 0:
            return None
        else:
            return idx, self.sections[idx]
            
    def read_doc(self):
        # form text chunks for llm to prompt over
        return self.sections
    
  