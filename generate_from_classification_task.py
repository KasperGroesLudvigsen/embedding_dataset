"""
Table 9: Prompt template for the long-short matching subgroup. For placeholders, “{num_words}” ∈ {"less than 10",
"at least 10", "at least 50", "at least 100", "at least 200"}, “{difficulty}” ∈ {high school, college, PhD}, “{clarity}” ∈
{clear, understandable with some effort, ambiguous}.
"""
from datasets import load_dataset
from generators import GenerateFromTextClassificationTask
import variables

def main():
    task_dataset_id = "synthetic-from-classification-tasks"

    language = "DANISH"
    num_words = ["less than 10", "at least 10", "at least 50", "at least 100", "at least 200"]
    difficulty = ["high school", "college", "PhD"]
    clarity = ["clear", "understandable with some effort", "ambiguous"]

    task = load_dataset(f"ThatsGroes/{variables.text_classification_task_dataset_name}-processed")
    task = list(task["train"]["response"])


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
        model_id=variables.model_id, 
        temperature=variables.temperature, 
        top_p=variables.top_p, 
        prompt=prompt, 
        language=variables.language,
        samples=variables.total_desired_samples,
        task=task,
        num_words=num_words,
        clarity=clarity,
        difficulty=difficulty
        )

    dataset = generator.generate()

    try:
        dataset.to_csv(f"{task_dataset_id}.csv")

    except Exception as e:

        print(f"could not save {variables.task_dataset_id}.csv")
        print(f"Exception: {e}")

    if variables.push_to_hf:
        dataset.push_to_hub(f"ThatsGroes/{task_dataset_id}")

if __name__ == "__main__":
    main()