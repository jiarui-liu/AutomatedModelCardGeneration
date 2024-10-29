# [Automatic Generation of Model and Data Cards: A Step Towards Responsible AI](https://arxiv.org/abs/2405.06258)

This paper has been accepted to NAACL 2024 Main as Oral.
- Arxiv: https://arxiv.org/abs/2405.06258
- ACL Anthhology: https://aclanthology.org/2024.naacl-long.110/

## Dataset

Our dataset is on HuggingFace: .

## Environment Setup

```bash
# install dependencies
conda create -n mc python=3.10
conda activate mc
pip install requirements.txt

# install the folder as a Python package named "modelcard" in editable mode
pip install -e .
```

## Reproduction

### Dataset Collection

In the folder `modelcard/data_collection`, `get_model_ids.py` crawls all the model ids sorted by the number of downloads in descending number. Below is an example script for running it.

```bash
# For model cards:
python3 modelcard/data_collection/get_model_ids.py --out_file data/model_info.jsonl --type model --start_idx 0 --end_idx 50

# For data cards:
python3 modelcard/data_collection/get_model_ids.py --out_file data/dataset_info.jsonl --type dataset --start_idx 0 --end_idx 50
```

`get_model_cards.py` then crawls the model cards on HuggingFace according to the collected model ids.

```bash
python3 modelcard/data_collection/get_model_cards.py --in_file data/model_info.jsonl --out_dir data/model_cards/ 
```

We further categorize the model cards into three classes:
1. Class A: Model card markdown files are not empty, and are validated by the HuggingFace team. We check if the strings "disclaimer" and "hugging face team" both exist in the model card to determine whether the model card is validated by HuggingFace. Model cards in this class will be the test set.
2. Class B: Other model cards that are not empty.
3. Class C: Empty model cards. We only consider class A and B for the later processing.

Next, we use `get_links.py` to find if a model card contains PDF paper links and/or GitHub README links, and get all the plausibile URLs. We prompt GPT-3.5-Turbo to validate direct source document links. The prompts are in `prompts.py`.

We use `spider.py` to download the paper PDFs and GitHub READMEs, and use [SciPDF parser](https://github.com/titipata/scipdf_parser) to convert PDF to text.

### Model Card Generation

In the folder `modelcard/gen_pipeline`, `model_card_generator.py` implements the functions needed for generating a whole model card. Several data directories and files are necessary to get it work:
1. `data/model_card2pdf.jsonl`: This file stores the paper links and hash of the stored PDF text file locations, using `direct_paper_link` and `direct_paper_hash`. You should hash the file locations on your own after you downloading PDFs and converting them to texts.
2. `data/total/scipdf/`: This folder stores the paper PDF texts. The file names are hashed.
3. `data/model_card2github_readme.jsonl`: Similarly, `direct_github_link` and `direct_github_hash` are what you need.
4. `data/total/github/`: This folder stores the crawled GitHub readmes. The file names are hashed.
5. `data/link_auto.jsonl` and `data/link_auto_github.jsonl`: These files store the necessary external contents to put in the generated model cards.

`modelcard/scripts` provides an example of calling `model_card_generator.py` using GPT-3.5-Turbo:
1. Run `gen_query_cache.py` to get sub-queries generated and cached. This procedure only needs to be run once.
2. Then run `gen_gpt3.5.py` to generate model cards for the models in the test set.

We use [VLLM](https://docs.vllm.ai/en/latest/getting_started/quickstart.html) to inference open-sourced LLMs using its OpenAI server-like encapsulation.

### Evaluation

- ROUGE-L: Run `eval/eval_rouge.py`.
- BERTScore: Clone the [bert_score](https://github.com/Tiiiger/bert_score) repo, download the required models, and run `eval/eval_bert_score.py`.
- BARTScore: Clone the [bart score](https://github.com/neulab/BARTScore) repo, download the required models, and run `eval/eval_bart_score.py`.
- NLI models: Run `eval/eval_nli.py`.
- Ragas: Clone the [Ragas](https://github.com/explodinggradients/ragas) repo and evaluate using `faithfulness`, `answer_relevancy`, `context_precision`, and `context_relevancy`.

## Citation

If you find our work useful, please give us a star and cite as follows :)

```
@inproceedings{liu-etal-2024-automatic,
    title = "Automatic Generation of Model and Data Cards: A Step Towards Responsible {AI}",
    author = "Liu, Jiarui  and
      Li, Wenkai  and
      Jin, Zhijing  and
      Diab, Mona",
    editor = "Duh, Kevin  and
      Gomez, Helena  and
      Bethard, Steven",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.naacl-long.110",
    doi = "10.18653/v1/2024.naacl-long.110",
    pages = "1975--1997",
    abstract = "In an era of model and data proliferation in machine learning/AI especially marked by the rapid advancement of open-sourced technologies, there arises a critical need for standardized consistent documentation. Our work addresses the information incompleteness in current human-written model and data cards. We propose an automated generation approach using Large Language Models (LLMs). Our key contributions include the establishment of CardBench, a comprehensive dataset aggregated from over 4.8k model cards and 1.4k data cards, coupled with the development of the CardGen pipeline comprising a two-step retrieval process. Our approach exhibits enhanced completeness, objectivity, and faithfulness in generated model and data cards, a significant step in responsible AI documentation practices ensuring better accountability and traceability.",
}
```