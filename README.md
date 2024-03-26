# llm_split_recall_test
A simple and efficient benchmark to evaluate in-context recall performance of Large Language Models (LLMs)

## Task Description
**Task Easy:** Sentence Split & Recall: for a given paragraph (the abstract section of the paper), split the paragraph into individual sentences
**Task Hard:** Sentence Split & Recall with Long Context: for a given paragraph (the abstract section of the paper) in a long document (determined by `hard_task_max_char`), split the paragraph into individual sentences

## Dataset
In the `eval_data.json` file, there are 10 papers from ACL 2023. File structure looks like following:

    [
    		{
    			"paper_id": int,
    			"paper_title": "paper title",
    			"paper_url": "link_to_the_paper",
    			"abstract_sentences": ["sentence1", "sentence2", ...],
    			"full_text": "full OCR text of the paper",
    		},
    		...
    ]

I have manually cleaned and splitted the abstracts into sentences.
The abstract in the `full_text` is also replaced with the cleaned sentences.

## Run the eval
 1. Clone the repo: `git clone https://github.com/ai8hyf/llm_split_recall_test`
 2. Install openai: `pip install openai`
 3. Provide your own openai api key or use your local LLM config. You can find the parameters inside `split_and_recall.py`. You can also choose the task (easy or hard) and context length for the hard task in the same file.
 4. Run the eval: `python split_and_recall.py`

## Preliminary Results
### Easy Task (hosted on vLLM, temp 0.1)
```markdown
| Model                    | Precision |
|--------------------------|-----------|
| Mistral 7B Instruct v0.2 |    61.04% |
| Mixtral 8x7B Instruct    |    87.01% |
| Qwen 1.5 72B Chat (4bit) |    96.10% |
| GPT-3.5-Turbo            |    97.40% |
| GPT-4-Turbo              |    98.70% |
```
### Hard Task (hosted on vLLM, temp 0.1)
```markdown
| Model                    | @ 2,500 Token | @ 5,000 Token | @ 8,000 Token |
|--------------------------|---------------|---------------|---------------|
| Mistral 7B Instruct v0.2 |         0.04% |            0% |            0% |
| Mixtral 8x7B Instruct    |        83.12% |            0% |            0% |
| Qwen 1.5 72B Chat (4bit) |        89.61% |        88.31% |        83.12% |
```

## Limitations
The eval data may not be representative.
Depending on the inference engine, prompt, and hyper-parameter settings, the benchmark scores may vary.

## Contributing
Please feel free to start new PRs!

## Citation
If you use Split&Recall in your research, please cite this project as:
```
@misc{splitNrecall,
  author = {Yifei Hu},
  title = {Split and Recall: A simple and efficient benchmark to evaluate in-context recall performance of Language Models},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/ai8hyf/llm_split_recall_test}},
}
```