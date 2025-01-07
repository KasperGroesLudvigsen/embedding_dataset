"""
Table 10: Prompt template for the short-short matching subgroup. We do not generate negative documents as the
matching task is already reasonably difficult.
"""

import random
from datasets import load_dataset, Dataset
import json
from vllm import LLM, SamplingParams
import pandas as pd 
from generators import GenerateFromTextMatchingTask

samples = 10
model_id = ""

temperature = 1.0
top_p = 1.0

task_dataset_id = "../.."

language = "DANISH"
task = ["task1", "task2"]
#task = load_dataset(task_dataset_id)
#task = list(task["train"]["response"])


prompt = f"""You have been assigned a text matching task: {{task}}
Your mission is to write one example for this task in JSON format. The JSON object must contain the following keys:
- "input": a string, a random input specified by the task.
- "positive_document": a string, a relevant document for the "input" according to the task.
Please adhere to the following guidelines:
- The values of all fields should be in {{language}}.
- Both the "input" and "positive_document" should be very short (a sentence or a phrase), avoid substantial word overlaps,
otherwise the task would be too easy.
- The "input" and "positive_document" should be independent of each other.
Your output must always be a JSON object only, do not explain yourself or output anything else. Be creative!"""

generator = GenerateFromTextMatchingTask(
    model_id=model_id, 
    temperature=temperature, 
    top_p=top_p, 
    prompt=prompt, 
    language=language,
    samples=samples,
    task=task,
    )

dataset = generator.generate(samples=samples)

dataset.to_csv("synth_data.csv")

#dataset.push_to_hub("ThatsGroes/some_name")




