"""
Table 8 - retrieval task
"""

from datasets import load_dataset
from generators import GenerateFromRetrievalTask
import variables

def main():
    task_dataset_id = "synthetic-from-retrieval-tasks"

    language = "DANISH"
    num_words = ["50", "100", "200", "300", "400", "500"]
    difficulty = ["high school", "college", "PhD"]
    clarity = ["clear", "understandable with some effort", "ambiguous"]
    task = ["task1", "task2"]
    query_length = ["less than 5 words", "5 to 15 words", "at least 10 words"]
    query_type = ["extremely long-tail", "long-tail", "common"]

    task = load_dataset(f"ThatsGroes/{variables.retrieval_task_dataset_name}-processed")
    task = list(task["train"]["response"])


    prompt = f"""You have been assigned a retrieval task: {{task}}
    Your mission is to write one text retrieval example for this task in JSON format. The JSON object must contain the following
    keys:
    - "user_query": a string, a random user search query specified by the retrieval task.
    - "positive_document": a string, a relevant document for the user query.
    - "hard_negative_document": a string, a hard negative document that only appears relevant to the query.
    Please adhere to the following guidelines:
    - The "user_query" should be {{query_type}}, {{query_length}}, {{clarity}}, and diverse in topic.
    - All documents must be created independent of the query. Avoid copying the query verbatim. It’s acceptable if some parts of
    the "positive_document" are not topically related to the query.
    - All documents should be at least {{num_words}} words long.
    - The "hard_negative_document" contains some useful information, but it should be less useful or comprehensive compared
    to the "positive_document".
    - Both the query and documents should be in {{language}}.
    - Do not provide any explanation in any document on why it is relevant or not relevant to the query.
    - Both the query and documents require {{difficulty}} level education to understand.
    Your output must always be a JSON object only, do not explain yourself or output anything else. Be creative!"""


    generator = GenerateFromRetrievalTask(
        model_id=variables.model_id, 
        temperature=variables.temperature, 
        top_p=variables.top_p, 
        prompt=prompt, 
        language=variables.language,
        samples=variables.total_desired_samples,
        task=task,
        clarity=clarity,
        num_words=num_words,
        difficulty=difficulty,
        query_type=query_type,
        query_length=query_length
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