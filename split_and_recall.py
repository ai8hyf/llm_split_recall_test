#!/usr/bin/env python
import argparse
import os
import json

from dotenv import load_dotenv
from openai import OpenAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_anthropic import ChatAnthropic
from langchain_fireworks import Fireworks
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_vertexai import VertexAI
from langchain_mistralai.chat_models import ChatMistralAI

import time

load_dotenv()


eval_task = "easy" # or choose "hard" and provide the max_char parameter
hard_task_max_char = 10000 # how many characters to provide to the model for the hard task

oai_api_key = os.getenv("OPENAI_API_KEY") # if you want to use the openai api
oai_model = "gpt-3.5-turbo" # if you want to use the openai api

local_llm_api_base = "http://localhost:8080/v1" # if you want to use the local model
local_llm_model = "path_to_your_local_model" # if you want to use the local model

model_temperature = 0.2 # feel free to tweak this parameter

MODEL_LIST = [
	"anthropic/claude-3-haiku-20240307",
	"anthropic/claude-3-opus-20240229",
	"anthropic/claude-3-sonnet-20240229",
	"fireworks/dbrx-instruct",
	"google/gemini-1.0-pro",
	"google/gemini-1.5-pro-latest",		# langchain cannot use generativeainot the right name, perhaps still in preview mode?
	"vertexai/gemini-1.5-pro-preview-0215",
	"mistral/mistral-large-latest",
	"openai/gpt-3.5-turbo",
	"openai/gpt-4-turbo",
	"vllm/mistral-7b-instruct-v0.2",
	"vllm/mistral-8x7b-instruct",
]

def load_eval_data(filename: str="eval_data.json") -> list[dict]:
	# eval data structure:
	# [
	# 	{
	# 		"paper_id": int,
	# 		"paper_title": "paper title",
	# 		"paper_url": "link_to_the_paper",
	# 		"abstract_sentences": ["sentence1", "sentence2", ...],
	# 		"full_text": "full OCR text of the paper",
	# 	},
	# 	...
	# ]
	# Note: 
	# 1. I have manually cleaned and splitted the abstracts into sentences. 
	# 2. The abstract in the full_text is also replaced with the cleaned sentences.
	# 3. Currently, there are 10 documents (papers) in the eval_data.json. They are from ACL 2023.
	with open(filename, 'r') as f:
		eval_data = json.load(f)
	
	return eval_data

def call_openai_api(prompt):
	client = OpenAI(
		# base_url=local_llm_api_base, # uncomment this line if you want to use the local model
		api_key=oai_api_key, # change to api_key=local_llm_api_key if you want to use the local model
	)
	completion = client.chat.completions.create(
		model= oai_model, # change to model=local_llm_model if you want to use the local model
		messages=[
			{"role": "system", "content": "You are a helpful assistant."}, # for mixtral models, you can comment out the system message
			{"role": "user", "content": prompt},
		],
		# max_tokens = 500, # you can uncomment this line if you want to limit the tokens
		temperature = model_temperature,
	)
	
	return completion.choices[0].message.content

def build_prompt_template() -> ChatPromptTemplate:
	return ChatPromptTemplate.from_messages(
		[
			("system", "You are a helpful assistant."),
			("user", "{prompt}"),
		]
	)

def call_langchain_sdk(prompt: str, model: str):

	provider, model_name = model.split('/', 1)

	if provider == "anthropic":
		model = ChatAnthropic(model=model_name, temperature=model_temperature)
	elif provider == "google":
		model = GoogleGenerativeAI(model=model_name, temperature=model_temperature)
	elif provider == "vertexai":
		model = VertexAI(
			model=model_name,
			project=os.getenv("GCP_PROJECT"),
			location=os.getenv("GCP_LOCATION"),
			temperature=model_temperature
		)
	elif provider == "mistral":
		model = ChatMistralAI(model=model_name, temperature=model_temperature)
	elif provider == "fireworks":
		# TODO: this is an instruct model, so will likely need different prompt.
		model = Fireworks(
			model=f"accounts/fireworks/models/{model_name}",
			base_url="https://api.fireworks.ai/inference/v1/completions",
			temperature=model_temperature
		)

	prompt_template = build_prompt_template()
	output_parser = StrOutputParser()
	chain = prompt_template | model | output_parser

	return chain.invoke({"prompt": prompt})

def build_easy_task_instruction(sentences):
	instruction = "Below is the abstract section from an academic paper.\n\n"

	instruction += "### The abstract: "
	instruction += " ".join(sentences)
	instruction += "\n\n"

	instruction += "Your task is to split the abstract into sentences. You answer can only include the sentences. Split them with newline symbol. Start each sentence with # symble. Do not try to modify the sentences. Do not add anything before or after the sentences.\n\n"
	instruction += "### Your answer: \n"

	return instruction

def build_hard_task_instruction(full_text, max_char):
	instruction = "Below is a part of an OCR result of an academic paper.\n\n"

	instruction += "### The OCR result: \n"
	instruction += full_text[:max_char]
	instruction += "\n### END OF OCR RESULT\n\n"

	instruction += "Your task is to split the abstract section into sentences. You answer can only include the sentences. Split them with newline symbol. Start each sentence with # symble. Do not try to modify the sentences. Do not add anything before or after the sentences.\n\n"
	instruction += "### Your answer: \n"

	return instruction

def evaluate_easy_task(*, eval_data: list[dict], model: str):
	total_sentence_count = 0
	total_correct_sentence_count = 0
	total_document_count = len(eval_data)
	processed_document_count = 0

	print("=== Easy task eval starts ===")
	print(f"model: {model}")

	for doc in eval_data:
		instruction = build_easy_task_instruction(doc['abstract_sentences'])

		if model.split('/')[0] in ["openai", "vllm"]:
			response = call_openai_api(instruction)
		else:
			response = call_langchain_sdk(instruction, model)
			if model.split('/')[0] in ["vertexai"]:
				time.sleep(12)

		generated_sentences = response.split("\n")

		total_sentence_count += len(doc['abstract_sentences'])

		document_correct_sentence_count = 0

		processed_document_count += 1
		
		for sentence in generated_sentences:
			sentence = sentence.strip()
			if sentence.startswith("#"):
				sentence = sentence[1:].strip()
			if sentence in doc['abstract_sentences']:
				total_correct_sentence_count += 1
				document_correct_sentence_count += 1

		print(f"Document: {doc['paper_url']}")
		print("Success rate: ", str(document_correct_sentence_count) + "/" + str(len(doc['abstract_sentences'])))
		print("=== " + str(processed_document_count) + "/" + str(total_document_count) + " documents processed ===")

	print("=== Easy task eval result: ===")
	print("Total success rate: ", str(total_correct_sentence_count) + "/" + str(total_sentence_count) + " = " + str(total_correct_sentence_count / total_sentence_count))

def evaluate_hard_task(*, eval_data: list[dict], model: str):
	total_sentence_count = 0
	total_correct_sentence_count = 0
	total_document_count = len(eval_data)
	processed_document_count = 0

	print("=== Hard task eval starts ===")

	for doc in eval_data:
		instruction = build_hard_task_instruction(doc['full_text'], hard_task_max_char)
		
		if model.split('/', 1)[0] in ["openai", "vllm"]:
			response = call_openai_api(instruction)
		else:
			response = call_langchain_sdk(instruction, model)

		generated_sentences = response.split("\n")

		total_sentence_count += len(doc['abstract_sentences'])

		document_correct_sentence_count = 0

		processed_document_count += 1
		
		for sentence in generated_sentences:
			sentence = sentence.strip()
			if sentence.startswith("#"):
				sentence = sentence[1:].strip()
			if sentence in doc['abstract_sentences']:
				total_correct_sentence_count += 1
				document_correct_sentence_count += 1

		print(f"Document: {doc['paper_url']}")
		print("Success rate: ", str(document_correct_sentence_count) + "/" + str(len(doc['abstract_sentences'])))
		print("=== " + str(processed_document_count) + "/" + str(total_document_count) + " documents processed ===")

	print("=== Hard task eval result: ===")
	print("Total success rate: ", str(total_correct_sentence_count) + "/" + str(total_sentence_count) + " = " + str(total_correct_sentence_count / total_sentence_count))


def main(args: argparse.Namespace):

	eval_task = args.eval_task
	model = args.model

	eval_data = load_eval_data()

	if eval_task == "easy":
		evaluate_easy_task(eval_data=eval_data, model=model)
	elif eval_task == "hard":
		evaluate_hard_task(eval_data=eval_data, model=model)
	else:
		SystemExit("Invalid eval_task")


def setup_parser():
	parser = argparse.ArgumentParser(description='Split and Recall Evaluation')
	parser.add_argument(
		'--eval_task',
		choices=['easy', 'hard'],
		default='easy',
		help='Evaluation task (easy or hard)'
	)
	parser.add_argument(
		'--model',
		choices=MODEL_LIST,
		default='openai/gpt-4-turbo',
		help='List of models in the form of inference-service-provider/model-name'
	)
	return parser.parse_args()

if __name__ == "__main__":
    args = setup_parser()
    main(args)
