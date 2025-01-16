"""
Implements table 10 in https://arxiv.org/pdf/2401.00368
"""

from utils import generate_task
import variables

prompt = f"""Brainstorm a list of text matching tasks where the queries are long documents.
Here are a few examples:
- Given a document that supports a debatable argument, find another document that contains opposite arguments.
- Provided a lengthy business proposal, retrieve competitive business strategies in the same industry.
Your output must always be a python list of strings only, with about 20 elements, and each element corresponds to a distinct
task in one sentence. Do not explain yourself or output anything else. Be creative!"""


generate_task(total_desired_samples=variables.total_desired_samples,
              model_id=variables.model_id,
              prompt=prompt,
              csv_save_as=csv_save_as,
              push_to_hf=push_to_hf,
              hf_dataset_name=variables.text_matching_long_dataset_name)