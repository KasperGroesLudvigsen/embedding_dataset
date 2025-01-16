"""
Implements table 10 in https://arxiv.org/pdf/2401.00368
"""

from utils import generate_task
import variables

def main():
    prompt = f"""Brainstorm a list of text matching tasks where both the queries and the groundtruth documents are very short (one or two
    sentences, even a short phrase).
    Here are a few examples:
    - Given a scientific paper title, retrieve the title of papers that cite the given paper.
    - Match a word with its definition.
    - Provided a notable person’s name, identify their occupation or achievement.
    Your output must always be a python list of strings only, with about 20 elements, and each element corresponds to a distinct
    task in one sentence. Do not explain yourself or output anything else. Be creative!"""

    csv_save_as = f"{variables.text_matching_short_dataset_name}.csv"
    push_to_hf = True

    generate_task(total_desired_samples=variables.total_desired_samples,
                model_id=variables.model_id,
                prompt=prompt,
                csv_save_as=csv_save_as,
                push_to_hf=push_to_hf,
                hf_dataset_name=variables.text_matching_short_dataset_name)
    
if __name__ == "__main__":
    main()