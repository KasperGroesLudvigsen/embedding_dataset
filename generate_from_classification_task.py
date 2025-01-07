"""
Table 9: Prompt template for the long-short matching subgroup. For placeholders, “{num_words}” ∈ {"less than 10",
"at least 10", "at least 50", "at least 100", "at least 200"}, “{difficulty}” ∈ {high school, college, PhD}, “{clarity}” ∈
{clear, understandable with some effort, ambiguous}.
"""

import random
from datasets import load_dataset, Dataset
import json
from vllm import LLM, SamplingParams
import pandas as pd 
from generators import GenerateFromTextClassificationTask

samples = 10
model_id = ""

temperature = 1.0
top_p = 1.0

task_dataset_id = "../.."

language = "DANISH"
num_words = ["less than 10", "at least 10", "at least 50", "at least 100", "at least 200"]
difficulty = ["high school", "college", "PhD"]
clarity = ["clear", "understandable with some effort", "ambiguous"]
task = ["task1", "task2"]
#task = load_dataset(task_dataset_id)
#task = list(task["train"]["response"])


prompt = f"""You have been assigned a text classification task: {{task}}

Your mission is to write one text classification example for this task in JSON format. The JSON object must contain the following keys:
- "input_text": a string, the input text specified by the classification task.
- "label": a string, the correct label of the input text.
- "misleading_label": a string, an incorrect label that is related to the task.

Please adhere to the following guidelines:
- The "input_text" should be {{num_words}} words and diverse in expression.
- The "misleading_label" must be a valid label for the given task, but not as appropriate as the "label" for the
"input_text".
- The values for all fields should be in {{language}}.
- Avoid including the values of the "label" and "misleading_label" fields in the "input_text", that would make
the task too easy.
- The "input_text" is {{clarity}} and requires {{difficulty}} level education to comprehend.

Your output must always be a JSON object only, do not explain yourself or output anything else. Be creative!"""

generator = GenerateFromTextClassificationTask(
    model_id=model_id, 
    temperature=temperature, 
    top_p=top_p, 
    prompt=prompt, 
    language=language,
    samples=samples
    )

dataset = generator.generate(samples=samples)

dataset.to_csv("synth_data.csv")

#dataset.push_to_hub("ThatsGroes/some_name")




