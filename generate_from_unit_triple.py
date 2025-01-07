"""
Table 10: Prompt template for the short-short matching subgroup. We do not generate negative documents as the
matching task is already reasonably difficult.
"""

import random
from datasets import load_dataset, Dataset
import json
from vllm import LLM, SamplingParams
import pandas as pd 
from generators import GenerateUnitTriple

samples = 10
model_id = ""

temperature = 1.0
top_p = 1.0

task_dataset_id = "../.."

language = "DANISH"
task = ["task1", "task2"]
#task = load_dataset(task_dataset_id)
#task = list(task["train"]["response"])
unit = ["sentence", "phrase", "passage"]
high_score = ["4", "4.5", "5"]
low_score = ["2.5", "3", "3.5"]
difficulty = ["elementary school", "high school", "college"]

prompt = f"""Write a {{unit}} triple with varying semantic similarity scores in JSON format. The semantic similarity score ranges from 1 to
5, with 1 denotes least similar and 5 denotes most similar.
Please adhere to the following guidelines:
- The keys in JSON are "S1", "S2", and "S3", the values are all strings in {{language}}, do not add any other keys.
- There should be some word overlaps between all three {{unit}}s.
- The similarity score between S1 and S2 should be {{high_score}}.
- The similarity score between S1 and S3 should be {{low_score}}.
- The {{unit}}s require {{difficulty}} level education to understand and should be diverse in terms of topic and length.
Your output must always be a JSON object only with three keys "S1", "S2" and "S3", do not explain yourself or output
anything else. Be creative!"""

generator = GenerateUnitTriple(
    model_id=model_id, 
    temperature=temperature, 
    top_p=top_p, 
    prompt=prompt, 
    language=language,
    task=task,
    samples=samples,
    unit=unit
    )

dataset = generator.generate(samples=samples)

dataset.to_csv("synth_data.csv")

#dataset.push_to_hub("ThatsGroes/some_name")




