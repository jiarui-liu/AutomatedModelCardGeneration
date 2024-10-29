messages_validating_github_links = [
    {"role": "system", "content": "You are a helpful assistant."},
    # add few shot prompts here to improve the performance.
    {"role": "user", "content": """Descriptions about the model {model} that might contain its github link are enclosed by ``` below:
```
{ref}
```

Here are some candidate github link references from the passage above:
{paper_link}
Which github link should be the direct code repo of the model? If none of the papers are the direct github link, please answer "None"."""}    
]

messages_validating_paper_links = [
    {"role": "system", "content": "You are a helpful assistant."},
    # add few shot prompts here to improve the performance.
    {"role": "user", "content": """Descriptions about the model {model} that might contain its paper links are enclosed by ``` below:
```
{ref}
```

Here are some candidate paper link references from the passage above:
{paper_link}
Which paper should be the direct one that introduces the model? If none of the papers are the direct reference to the model, please answer "None"."""}
]