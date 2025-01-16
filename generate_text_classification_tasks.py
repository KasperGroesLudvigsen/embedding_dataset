"""
Implements table 9 in https://arxiv.org/pdf/2401.00368
"""

from utils import generate_task
import variables

prompt = f"""Brainstorm a list of potentially useful text classification tasks.
Please adhere to the following guidelines:
- Tasks should cover a diverse range of domains and task types.
Your output must always be a Python list of strings only, with about 20 elements, and each element corresponds to a distinct text classification task in one sentence. Do not explain yourself or output anything else. Be creative!"""

csv_save_as = f"{variables.text_classification_task_dataset_name}.csv"
push_to_hf = True

generate_task(total_desired_samples=variables.total_desired_samples,
              model_id=variables.model_id,
              prompt=prompt,
              csv_save_as=csv_save_as,
              push_to_hf=push_to_hf,
              hf_dataset_name=variables.text_classification_task_dataset_name)