import re
from langchain.schema import AIMessage, HumanMessage, SystemMessage

class Template:
    header = '''Imagine that you are in a deep learning model development team. You are ready to publish your model to Huggingface, and you need to write the model card description. In your team, you work as {role}. Please use the contents from the following "References" bullet point as the context to answer the question required below about your model at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.'''
    
    header = '''Imagine that you are in a deep learning model development team. You are ready to publish your model to Huggingface, and you need to write the model card description. In your team, you work as {role}. Please use the contents from the following "References" bullet point as the context to answer the question required below about your model at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Please include links occurred in the references if necessary.'''
    
    header = '''Imagine that you are in a deep learning model development team. You are ready to publish your model to Huggingface, and you need to write the model card description. In your team, you work as {role}.'''
    
    roles = {
        'developer': 'the developer who writes the code and runs training',
        'sociotechnic': 'the sociotechnic who is skilled at analyzing the interaction of technology and society long-term (this includes lawyers, ethicists, sociologists, or rights advocates)',
        'project_organizer': 'the project organizer who understands the overall scope and reach of the model and can roughly fill out each part of the card, and who serves as a contact person for model card updates'
    }
    
    default_answer = "[More Information Needed]"
    
    def get_division_prompt(self, query):
        prompt = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=f"""Below is the question that you need to answer about a model given its paper document and github repo readme file. If the question includes multiple sub-problems, provide the python list of subproblem strings that you think are more atomic and easy to search for on the source documents. Otherwise, just output "False".

```
Provide basic details about the model. This mainly includes the architecture. Training procedures, parameters, and important disclaimers can also be mentioned in this section.
```

Your answer should follow the format below if it can be divided into sub-problems:
```
["sub_problem_1", "sub_problem_2", "sub_problem_3"]
```
"""
            ),
            AIMessage(content=f"""["Provide basic details about the model architecture of the model.", "Provide basic details about the training procedure of the model.", "Provide basic details about the parameters of the model.", "Provide basic details about the important disclaimers of the model."]"""),
            HumanMessage(content=f"""Below is the question that you need to answer about a model given its paper document and github repo readme file. If the question includes multiple sub-problems, provide the python list of subproblem strings that you think are more atomic and easy to search for on the source documents. Otherwise, just output "False".

```
Provide a 1-2 sentence summary of what the model is.
```

Your answer should follow the format below if it can be divided into sub-problems:
```
["sub_problem_1", "sub_problem_2", "sub_problem_3"]
```
"""
            ),
            AIMessage(content="""False"""),
            HumanMessage(content=f"""Below is the question that you need to answer about a model given its paper document and github repo readme file. If the question includes multiple sub-problems, provide the python list of subproblem strings that you think are more atomic and easy to search for on the source documents. Otherwise, just output "False".

```
{query}
```

Your answer should follow the format below if it can be divided into sub-problems:
```
["sub_problem_1", "sub_problem_2", "sub_problem_3"]
```
"""
            ),
        ]
        return prompt

    def get_pseudo_answer_prompt(self, query, model=None):
        if model is not None:
            prompt = [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content=f"Please write a short passage to answer the question about the model {model}: {query}")
            ]
        else:
            # ablation study: uniform pseudo answer for all models
            prompt = [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content=f"Please write a short passage to answer the question about a huggingface model: {query}")
            ]
        return prompt
    
    def get_section_retrieval_prompt(self, query, model, paper_sections: list | None, github_sections: list | None):
        def format_paper_section_string(paper_sections):
            if paper_sections is None:
                return ""
            bullet_str = "\n".join([f"- {sec}" for sec in paper_sections])
            return f"""The paper of the model includes the following sections:
```
{bullet_str}
```

"""     
        def format_github_section_string(github_sections):
            if github_sections is None:
                return ""
            bullet_str = "\n".join([f"- {sec}" for sec in github_sections])
            return f"""The github repo of the model includes the following sections:
```
{bullet_str}
```

"""  

        prompt = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=f"""Here are a list of paper sections and github repo readme sections for the model {model}:
{format_paper_section_string(paper_sections)}{format_github_section_string(github_sections)}"""),
            AIMessage(content="Hi, how can I help you with that information about the model?"),
            HumanMessage(content=f"""According to section names above, which section(s) of the paper and the github readme do you think may contain the relevant information to the following question:
```
Provide a 1-2 sentence summary of what the model {model} is.
```

Select top **three** sections that you think are the most relevant to the question. Your answer should follow the format below:
```
- <section 1>
- <section 2>
- ...
```
"""),
            AIMessage(content=f"""
- title
- abstract
{"- " + github_sections[0] if github_sections is not None else ""}
"""),
            HumanMessage(content=f"""According to section names above, which section(s) of the paper and the github readme do you think may contain the relevant information to the following question:
```
{query}
```

Select top **three** sections that you think are the most relevant to the question. Your answer should follow the format below:
```
- <section 1>
- <section 2>
- ...
```
"""
            ),
        ]
        return prompt
    
    
    def get_generation_prompt(self, query, model, reference, role):
        prompt = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=f"""{self.header.format(role = self.roles.get(role, 'project_organizer'))}

Below is the reference to refer to and the question you need to answer for the model {model} that you have worked on:

References:
```
{reference}
```

Question:
```
{query}
```

Please refer to the above contents of "References" as the knowledge source to answer the question about your model {model}. If you don't know the answer for a specific part, you should just say "[More Information Needed]". You can write code only if you find a direct code block reference from above, otherwise just output "[More Information Needed]". Your answer should be easy to read and succinct.
"""),
        ]
        return prompt
    
def citation_bibtex_generator(title, authors, url):
    def build_author_list(authors):
        if authors is None:
            return "Author"
        authors = [i.strip() for i in authors.split(";")]
        authors = ' and\n              '.join([i for i in authors])
        return authors
    def build_tag(title, authors):
        tag = ""
        if authors is None:
            tag += "author"
        else:
            authors = authors.split(" ")
            if len(authors) == 0:
                tag += "author"
            else:
                tag += "".join(re.findall("[a-zA-Z]+", authors[0].lower()))
        
        tag += "-"
        if title is None:
            tag += "title"
        else:
            title = title.split(" ")
            if len(title) == 0:
                tag += "title"
            else:
                tag += "".join(re.findall("[a-zA-Z]+", title[0].lower()))
        tag += ","
        return tag
    return """```
@misc{{{tag}
    author = {{{author}}},
    title  = {{{title}}},
    url    = {{{url}}}
}}
```""".format(
    tag = build_tag(title, authors),
    author = build_author_list(authors),
    title = title,
    url = url,
)
        

class QuestionTemplate:
    
    summary = {
        'prompt': 'Provide a 1-2 sentence summary of what the model{model} is.',
        'role': 'project_organizer',
        'markdown_header': '# Model Card for {model}\n\n{answer}\n\n',
        'source': ['paper', 'github'],
        'keyword': ['summary', 'title', 'abstract'],
        'generation_chain_config': {
            "temperature": 0,
            "max_tokens": 256,
        }
    }
    
    description = {
        'prompt': 'Provide basic details about the model{model}. This includes the model architecture, training procedures, parameters, and important disclaimers.',
        'role': 'project_organizer',
        'markdown_header': '## Model Details\n\n### Model Description\n\n{answer}\n\n',
        'source': ['paper', 'github'],
        'keyword': ['architecture', 'model', 'method', 'train', 'parameter', 'disclaimer'],
        'generation_chain_config': {
            "temperature": 0,
            "max_tokens": 512,
        }
    }
    
    developed_by = {
        'prompt': 'special',
        'markdown_header': '- **Developed by:** {answer}\n',
    }
    
    funded_by = {
        'prompt': 'List the people or organizations that fund this project of the model{model}.',
        'role': 'project_organizer',
        'markdown_header': '- **Funded by:** {answer}\n',
        'source': ['paper'],
        'keyword': ['fund', 'acknowledgement', 'organization'],
        'generation_chain_config': {
            "temperature": 0,
            "max_tokens": 64,
        }
    }
    
    shared_by = {
        'prompt': 'Who are the contributors that made the model{model} available online as a GitHub repo?',
        'role': 'developer',
        'markdown_header': '- **Shared by:** {answer}\n',
        'source': ['paper', 'github'],
        'keyword': ['contributor', 'contribution', 'developer', 'contact', 'author'],
        'generation_chain_config': {
            "temperature": 0,
            "max_tokens": 64,
        }
    }
    
    model_type = {
        'prompt': 'Summarize the type of the model{model} in terms of the training method, machine learning type, and modality in one sentence.',
        'role': 'project_organizer',
        'markdown_header': '- **Model type:** {answer}\n',
        'source': ['paper', 'github'],
        'keyword': ['type', 'train', 'method', 'model'],
        'generation_chain_config': {
            "temperature": 0,
            "max_tokens": 64,
        }
    }
    
    language = {
        'prompt': 'Summarize what natural human language the model{model} uses or processes in one sentence.',
        'role': 'project_organizer',
        'markdown_header': '- **Language(s):** {answer}\n',
        'source': ['paper', 'github'],
        'keyword': ['language', 'lingual'],
        'generation_chain_config': {
            "temperature": 0,
            "max_tokens": 64,
        }
    }
    
    license = {
        'prompt': 'Provide the name and link to the license being used for the model{model}.',
        'role': 'project_organizer',
        'markdown_header': '- **License:** {answer}\n',
        'source': ['github'],
        'keyword': ['license'],
        'generation_chain_config': {
            "temperature": 0,
            "max_tokens": 32,
        }
    }
    
    finetuned_from = {
        'prompt': 'If the model{model} is fine-tuned from another model, provide the name and link to that base model.',
        'role': 'project_organizer',
        'markdown_header': '- **Finetuned from model:** {answer}\n',
        'source': ['paper', 'github'],
        'keyword': ['fine-tune', 'fine-tuning', 'tune'],
        'generation_chain_config': {
            "temperature": 0,
            "max_tokens": 128,
        }
    }
    
    model_sources = {
        'prompt': 'special',
        'markdown_header': '### Model Sources\n\n- **Repository:** {github_link}\n- **Paper:** {paper_link}\n',
    }
    
    demo_sources = {
        'prompt': 'Provide the link to the demo of the model{model}.',
        'role': 'project_organizer',
        'markdown_header': '- **Demo:** {answer}\n',
        'source': ['paper', 'github'],
        'keyword': ['demo', 'try'],
        'generation_chain_config': {
            "temperature": 0,
            "max_tokens": 32,
        }
    }
    
    direct_use = {
        'prompt': 'Explain how the model{model} can be used without fine-tuning, post-processing, or plugging into a pipeline. Provide a code snippet if necessary',
        'role': 'project_organizer',
        'markdown_header': '## Uses\n\n### Direct Use\n\n{answer}\n\n',
        'source': ['paper', 'github'],
        'keyword': ['direct', 'use', 'usage', 'quickstart'],
        'generation_chain_config': {
            "temperature": 0,
            "max_tokens": 512,
        }
    }
    
    downstream_use = {
        'prompt': 'Explain how this model{model} can be used when fine-tuned for a task or when plugged into a larger ecosystem or app. Provide a code snippet if necessary',
        'role': 'project_organizer',
        'markdown_header': '### Downstream Use\n\n{answer}\n\n',
        'source': ['paper', 'github'],
        'keyword': ['downstream', 'use', 'usage', 'fine-tune', 'fine-tuning', 'tune'],
        'generation_chain_config': {
            "temperature": 0,
            "max_tokens": 512,
        }
    }
    
    out_of_scope_use = {
        'prompt': 'How the model may foreseeably be misused and address what users ought not do with the model{model}.',
        'role': 'sociotechnic',
        'markdown_header': '### Out-of-Scope Use\n\n{answer}\n\n',
        'source': ['paper', 'github'],
        'keyword': ['scope', 'misuse', 'out-of-scope'],
        'generation_chain_config': {
            "temperature": 0,
            "max_tokens": 512,
        }
    }
    
    bias_risks_limitations = {
        'prompt': 'What are the known or foreseeable issues stemming from this model{model}? These include foreseeable harms, misunderstandings, and technical and sociotechnical limitations.',
        'role': 'sociotechnic',
        'markdown_header': '### Bias, Risks, and Limitations\n\n{answer}\n\n',
        'source': ['paper', 'github'],
        'keyword': ['bias', 'risk', 'limitation', 'issue', 'ethic', 'moral'],
        'generation_chain_config': {
            "temperature": 0,
            "max_tokens": 512,
        }
    }
    
    bias_recommendations = {
        'prompt': 'What are recommendations with respect to the foreseeable issues about the model{model}?',
        'role': 'sociotechnic',
        'markdown_header': '### Recommendations\n\n{answer}\n\n',
        'source': ['paper', 'github'],
        'keyword': ['bias', 'risk', 'limitation', 'issue', 'ethic', 'moral', 'recommendation'],
        'generation_chain_config': {
            "temperature": 0,
            "max_tokens": 512,
        }
    }
    
    training_data = {
        'prompt': 'Write 1-2 sentences on what the training data of the model{model} is. Links to documentation related to data pre-processing or additional filtering may go here as well as in More Information.',
        'role': 'developer',
        'markdown_header': '## Training Details\n\n### Training Data\n\n{answer}\n\n',
        'source': ['paper', 'github'],
        'keyword': ['data', 'pre process', 'pre-process', 'training data'],
        'generation_chain_config': {
            "temperature": 0,
            "max_tokens": 128,
        }
    }
    
    preprocessing = {
        'prompt': 'Provide detail tokenization, resizing/rewriting (depending on the modality), etc. about the preprocessing for the data of the model{model}.',
        'role': 'developer',
        'markdown_header': '### Training Procedure\n\n#### Preprocessing\n\n{answer}\n\n',
        'source': ['paper', 'github'],
        'keyword': ['tokenization', 'data', 'pre process', 'pre-process', 'training data'],
        'generation_chain_config': {
            "temperature": 0,
            "max_tokens": 512,
        }
    }
    
    training_regime = {
        'prompt': 'Provide detail training hyperparameters when training the model{model}.',
        'role': 'developer',
        'markdown_header': '#### Training Hyperparameters\n\n{answer}\n\n',
        'source': ['paper', 'github'],
        'keyword': ['hyperparameter', 'hyper-parameter', 'training detail', 'parameter', 'train'],
        'generation_chain_config': {
            "temperature": 0,
            "max_tokens": 512,
        }
    }
    
    speeds_sizes_times = {
        'prompt': 'Provide detail throughput, start or end time, checkpoint sizes, etc. about the model{model}.',
        'role': 'developer',
        'markdown_header': '#### Speeds, Sizes, Times\n\n{answer}\n\n',
        'source': ['paper', 'github'],
        'keyword': ['model', 'experiment', 'time', 'checkpoint'],
        'generation_chain_config': {
            "temperature": 0,
            "max_tokens": 512,
        }
    }
    
    testing_data = {
        'prompt': 'Provide benchmarks or datasets that the model{model} evaluates on.',
        'role': 'developer',
        'markdown_header': '## Evaluation\n\n### Testing Data, Factors & Metrics\n\n#### Testing Data\n\n{answer}\n\n',
        'source': ['paper', 'github'],
        'keyword': ['evaluation', 'data', 'test', 'dataset', 'benchmark'],
        'generation_chain_config': {
            "temperature": 0,
            "max_tokens": 256,
        }
    }
    
    testing_factors = {
        'prompt': 'What are the foreseeable characteristics that will influence how the model{model} behaves? This includes domain and context, as well as population subgroups. Evaluation should ideally be disaggregated across factors in order to uncover disparities in performance.',
        'role': 'sociotechnic',
        'markdown_header': '#### Factors\n\n{answer}\n\n',
        'source': ['paper', 'github'],
        'keyword': ['discussion', 'bias', 'risk', 'limitation', 'issue', 'et,hic', 'moral', 'evaluation', 'data', 'test', 'dataset', 'benchmark', 'factor'],
        'generation_chain_config': {
            "temperature": 0,
            "max_tokens": 512,
        }
    }
    
    testing_metrics = {
        'prompt': 'What metrics will be used for evaluation in light of tradeoffs between different errors about the model{model}?',
        'role': 'developer',
        'markdown_header': '#### Metrics\n\n{answer}\n\n',
        'source': ['paper', 'github'],
        'keyword': ['metric', 'evaluation', 'test'],
        'generation_chain_config': {
            "temperature": 0,
            "max_tokens": 512,
        }
    }
    
    results = {
        'prompt': 'Provide evaluation results of the model{model} based on the Factors and Metrics.',
        'role': 'developer',
        'markdown_header': '### Results\n\n{answer}\n\n',
        'source': ['paper', 'github'],
        'keyword': ['result', 'evaluation', 'factor', 'metric', 'test'],
        'generation_chain_config': {
            "temperature": 0,
            "max_tokens": 512,
        }
    }
    
    results_summary = {
        'prompt': 'Summarize the evaluation results about the model{model}.',
        'role': 'developer',
        'markdown_header': '#### Summary\n\n{answer}\n\n',
        'source': ['paper', 'github'],
        'keyword': ['result', 'evaluation', 'factor', 'metric', 'test'],
        'generation_chain_config': {
            "temperature": 0,
            "max_tokens": 256,
        }
    }
    
    model_examination = {
        'prompt': 'This is an experimental section some developers are beginning to add, where work on explainability/interpretability may go about the model{model}.',
        'role': 'developer',
        'markdown_header': '## Model Examination\n\n{answer}\n\n',
        'source': ['paper', 'github'],
        'keyword': ['discussion', 'result', 'interpret', 'explain'],
        'generation_chain_config': {
            "temperature": 0,
            "max_tokens": 512,
        }
    }
    
    hardware = {
        'prompt': 'Provide the hardware type that the model{model} is trained on.',
        'role': 'developer',
        'markdown_header': '## Environmental Impact\n\n- **Hardware Type:** {answer}\n',
        'source': ['paper', 'github'],
        'keyword': ['hardware', 'training detail', 'environment', 'train', 'infrastructure'],
        'generation_chain_config': {
            "temperature": 0,
            "max_tokens": 32,
        }
    }
    
    software = {
        'prompt': 'Provide the software type that the model{model} is trained on.',
        'role': 'developer',
        'markdown_header': '- **Software Type:** {answer}\n',
        'source': ['paper', 'github'],
        'keyword': ['software', 'training detail', 'environment', 'train', 'infrastructure'],
        'generation_chain_config': {
            "temperature": 0,
            "max_tokens": 32,
        }
    }
    
    hours_used = {
        'prompt': 'Provide the amount of time used to train the model{model}.',
        'role': 'developer',
        'markdown_header': '- **Hours used:** {answer}\n',
        'source': ['paper', 'github'],
        'keyword': ['time', 'training detail', 'environment', 'train', 'infrastructure'],
        'generation_chain_config': {
            "temperature": 0,
            "max_tokens": 32,
        }
    }
    
    cloud_provider = {
        'prompt': 'Provide the cloud provider that the model{model} is trained on.',
        'role': 'developer',
        'markdown_header': '- **Cloud Provider:** {answer}\n',
        'source': ['paper', 'github'],
        'keyword': ['cloud provider', 'training detail', 'environment', 'train', 'infrastructure'],
        'generation_chain_config': {
            "temperature": 0,
            "max_tokens": 32,
        }
    }
    
    # cloud_region = {}
    
    co2_emitted = {
        'prompt': 'Provide the amount of carbon emitted when training the model{model}.',
        'role': 'developer',
        'markdown_header': '- **Carbon Emitted:** {answer}\n',
        'source': ['paper', 'github'],
        'keyword': ['carbon', 'training detail', 'environment', 'train', 'infrastructure'],
        'generation_chain_config': {
            "temperature": 0,
            "max_tokens": 32,
        }
    }
    
    model_specs = {
        'prompt': 'Provide the model architecture and objective about the model{model}.',
        'role': 'developer',
        'markdown_header': '## Technical Specification\n\n### Model Architecture and Objective\n\n{answer}\n\n',
        'source': ['paper', 'github'],
        'keyword': ['architecture', 'objective', 'model'],
        'generation_chain_config': {
            "temperature": 0,
            "max_tokens": 512,
        }
    }
    
    compute_infrastructure = {
        'prompt': 'Provide the compute infrastructure about the model{model}.',
        'role': 'developer',
        'markdown_header': '### Compute Infrastructure\n\n{answer}\n\n',
        'source': ['paper', 'github'],
        'keyword': ['training detail', 'environment', 'train', 'infrastructure'],
        'generation_chain_config': {
            "temperature": 0,
            "max_tokens": 256,
        }
    }
    
    citation_bibtex = {
        'prompt': 'special',
        'helper_func': citation_bibtex_generator,
        'markdown_header': '## Citation\n\n{answer}\n\n',
    }
    
    def set_attribute(self, **kwargs):
        """Modify existing attributes or add new attributes."""
        for key, value in kwargs.items():
            if key.startswith('__') or callable(getattr(self, key)):
                continue
            setattr(self, key, value)
    
    def get_mapping(self):
        """Get the mapping dictionary of all question templates."""
        mapping = {}
        for attr in dir(self):
            # Filter out special methods and attributes
            if not callable(getattr(self, attr)) and not attr.startswith("__"):
                # Map the string representation to the actual attribute
                mapping[attr] = getattr(self, attr)
        return mapping

    def list_variables(self):
        return [attr for attr in self.__class__.__dict__ if not callable(getattr(self, attr)) and not attr.startswith("__")]