"""
Implements table 8 in https://arxiv.org/pdf/2401.00368
"""

from utils import generate_task
import variables

def main():
    prompt = f"""Brainstorm a list of potentially useful text retrieval tasks.
    Here are a few examples for your reference:
    - Retrieve relevant documents for a short keyword web search query that asks for weather information.
    - Search for documents that answers a FAQ-style query on children’s nutrition.
    Please adhere to the following guidelines:
    - Specify what the query is, and what the desired documents are.
    - Each retrieval task should cover a wide range of queries, and should not be too specific.
    Your output must always be a python list of strings only, with about 20 elements, and each element corresponds to a distinct
    retrieval task in one sentence. Do not explain yourself or output anything else. Be creative!"""

    csv_save_as = f"{variables.retrieval_task_dataset_name}.csv"
    push_to_hf = True

    generate_task(total_desired_samples=variables.total_desired_tasks,
                model_id=variables.model_id,
                prompt=prompt,
                csv_save_as=csv_save_as,
                push_to_hf=push_to_hf,
                hf_dataset_name=variables.retrieval_task_dataset_name)
    
if __name__ == "__main__":
    main()  