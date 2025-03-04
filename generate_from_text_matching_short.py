"""
Table 10: Prompt template for the short-short matching subgroup. We do not generate negative documents as the
matching task is already reasonably difficult.
"""

from datasets import load_dataset
from generators import GenerateFromTextMatchingTask
import variables
import argparse

def main(language: str):

    print(f"Will generate data in: {language}")

    task_dataset_id = "synthetic-from-text-mathing-short-tasks"

    task = ["task1", "task2"]
    task = load_dataset(f"ThatsGroes/{variables.text_matching_short_dataset_name}-processed")
    task = list(task["train"]["response"])


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
        model_id=variables.model_id, 
        temperature=variables.temperature, 
        top_p=variables.top_p, 
        prompt=prompt, 
        language=language,
        samples=variables.total_desired_samples,
        task=task,
        )

    dataset = generator.generate()

    try:
        dataset.to_csv(f"{task_dataset_id}-{language.lower()}.csv", index=False)

    except Exception as e:

        print(f"could not save {task_dataset_id}.csv")
        print(f"Exception: {e}")

    if variables.push_to_hf:
        
        if "(" in language or ")" in language:
        
            language = language.split("(")[0].strip()
        
        dataset.push_to_hub(f"ThatsGroes/{task_dataset_id}-{language.lower()}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process a language argument.")

    parser.add_argument("language", type=str, help="The language of the generated data.")

    args = parser.parse_args()

    main(language=args.language)