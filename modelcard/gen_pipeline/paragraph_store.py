import json
import sys
import numpy as np
import pandas as pd
from langchain.text_splitter import TextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document

from modelcard.gen_pipeline.markdown_reader import MarkdownReader
from modelcard.gen_pipeline.faiss_customized import MyFAISS
from modelcard.utils import cosine_similarity

class DocProcessor():
    def __init__(self, args):
        self.args = args
        
        self.doc_paper = None
        self.doc_github = None
        
        self.doc_paper = []
        self.doc_paper.extend(
            self.read_doc(
                args.paper_path,
                self.args.paper_type,
            ) # type: ignore
        )
        
        self.doc_github = []
        self.doc_github.extend(
            self.read_doc(
                args.github_path,
                self.args.github_type,
            ) # type: ignore
        )
    
    def format_str(self, doc):
        return [{"heading": d['heading'].replace(r"\{|\}", ""), "content": d['content'].replace(r"\{|\}", "")} for d in doc]
    
    def read_doc(self, doc_path, doc_type):
        print(f"Loading the doc {doc_path}..")
        doc = None
        with open(doc_path, 'r') as file:
            if doc_type == 'json':
                doc = json.load(file)
                
                res = []
                # heading: xxx, content: xxx
                info = {
                    'heading': None,
                    'content': None,
                    'source': 'paper',
                }
                
                title = doc.get("title", '')
                if title != '':
                    info['heading'] = 'title'
                    info['content'] = title
                    res.append(info.copy())
                
                authors = doc.get("authors", '')
                if authors != '':
                    info['heading'] = 'authors'
                    info['content'] = authors
                    res.append(info.copy())
                
                abstract = doc.get('abstract', '')
                if abstract != '':
                    info['heading'] = 'abstract'
                    info['content'] = abstract
                    res.append(info.copy())
                
                for section in doc.get("sections", []):
                    heading = section.get("heading", "")
                    txt = section.get("text", "")
                    if heading != "" and txt != "":
                        info['heading'] = heading
                        info['content'] = txt
                        res.append(info.copy())
                return self.format_str(res)
                
            elif doc_type == 'txt':
                doc = file.read()
                return [doc]
            elif doc_type == 'markdown':
                doc = file.read()
                reader = MarkdownReader(doc=doc)
                doc = reader.read_doc()
                return self.format_str(doc)


class ParagraphStore():
    """_summary_
    
    Referencces:
    - https://python.langchain.com/docs/integrations/text_embedding/sentence_transformers
    
    """
    def __init__(self, args, doc_args):
        self.args = args
        self.doc_args = doc_args
        
        if self.doc_args.embedding_model_name in [
            'sentence-transformers/all-MiniLM-L6-v2',
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            'sentence-transformers/all-mpnet-base-v2',
            ]:
            from langchain.embeddings.huggingface import HuggingFaceEmbeddings
            self.embedding_model = HuggingFaceEmbeddings(model_name=self.doc_args.embedding_model_name.split("/")[-1])
        elif self.doc_args.embedding_model_name in [
            'jinaai/jina-embeddings-v2-base-en'
        ]:
            from modelcard.gen_pipeline.embeddings_customized import TransformerEmbeddings
            self.embedding_model = TransformerEmbeddings(model_name=self.doc_args.embedding_model_name)
        
        
        if not hasattr(doc_args, 'database_name'):
            self.doc_args.database_name = 'faiss'
        self.database_name = self.doc_args.database_name

        self.get_splitter()
    
    def get_splitter(self):
        if self.doc_args.splitter == 'recursive':
            self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.doc_args.chunk_size, chunk_overlap=self.doc_args.chunk_overlap)
        
    def load_doc(self):
        self.doc_processor = DocProcessor(self.args)
        self.documents = self.doc_processor.doc_paper + self.doc_processor.doc_github # type: ignore
        print(f"Length of documents: {len(self.documents)}")
        
        if self.database_name == 'faiss':
            self.splitted_documents = []
            for doc in self.documents:
                contents = self.text_splitter.split_documents([Document(page_content=doc['content'], metadata="")])
                for idx, c in enumerate(contents):
                    row = doc.copy()
                    row.pop("content", None)
                    row['chunk'] = idx
                    
                    self.splitted_documents.append(Document(page_content=c.page_content, metadata=row))
            print(f"Length of splitted documents: {len(self.splitted_documents)}")
            self.vec_store = MyFAISS.from_documents(self.splitted_documents, self.embedding_model)
    
    def get_section(self, section_name):
        for doc in self.documents:
            heading = doc.get('heading', "")
            if heading == section_name:
                return doc['content']
        
        return None
    
    def get_headers(self, source='paper'):
        """Get the header list from the source document. The argument source should either be "paper" or "github".
        """
        source_documents = None
        if source == 'paper':
            source_documents = self.doc_processor.doc_paper
        elif source == 'github':
            source_documents = self.doc_processor.doc_github
        
        headers = []
        for doc in source_documents: # type: ignore
            heading = doc.get('heading', "")
            if heading != "":
                headers.append(heading)
        return headers
    
    def search(self, query, **kwargs):
        # k=3, 
        if self.database_name == 'faiss':
            docs = self.vec_store.similarity_search_with_score(query, **kwargs)
            return docs
