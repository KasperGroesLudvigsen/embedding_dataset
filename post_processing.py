"""
Post processing of the generated tasks
"""

from datasets import load_dataset, Dataset
import variables

def split_responses(dataset, column="response"):
    new_entries = []
    for entry in dataset[column]:
        # Extract the content inside the Python list
        response_list = eval(entry.strip("```python").strip())  # Convert string representation to a Python list
        if isinstance(response_list, list):
            new_entries.extend([sentence.strip().strip('"') for sentence in response_list])
        else:
            print(f"Unexpected format: {entry}")
    return new_entries

def post_process_classification_tasks():
    data_id = f"ThatsGroes/{variables.text_classification_task_dataset_name}"
    
    data = load_dataset(data_id)

    data_list = list(data["train"]["response"])

    new_data = []

    for i in data_list:

        i = i.replace("```", "").replace("```python", "").replace("[", "").replace("]", "")

        i = i.strip("'").lstrip('- ').strip()

        i = i.replace("'", "")

        new_data.extend(i.split("\n"))

    new_data = [n for n in new_data if n]

    new_data = Dataset.from_dict({"response" : new_data})

    new_data.push_to_hub(f"{data_id}-processed")


def split_and_clean(entry):
    """
    Splits a single dataset entry into multiple cleaned entries.
    Removes newlines and leading hyphens.
    """
    sentences = entry.split("\n")  # Split the entry by newlines
    cleaned_sentences = [sentence.lstrip("-").strip() for sentence in sentences if sentence.strip()]  # Clean each sentence
    return cleaned_sentences

# Process the dataset
def process_dataset(dataset):
    """
    Processes the dataset by splitting and cleaning entries.
    """
    all_sentences = []
    for entry in dataset["response"]:
        all_sentences.extend(split_and_clean(entry))
    return all_sentences


def get_hf_id(s):
    return f"ThatsGroes/{s}"

def main():

    post_process_classification_tasks()

    # classification tasks omitted as the dataset was properly formatted by gemma
    dataset_ids = [get_hf_id(variables.retrieval_task_dataset_name), get_hf_id(variables.text_matching_long_dataset_name), get_hf_id(variables.text_matching_short_dataset_name)]

    for id in dataset_ids:
        # Load your dataset
        data = load_dataset(id)  # Replace with your dataset name
        # Process the 'train' split
        train_sentences = process_dataset(data["train"])

        # Create a new Dataset from the cleaned sentences
        new_train_dataset = Dataset.from_dict({"response": train_sentences})

        new_train_dataset.push_to_hub(f"{id}-processed")
        # Replace the 'train' split with the new dataset
        #processed_data = DatasetDict({"train": new_train_dataset})

        # Save or inspect the processed dataset
        #processed_data.save_to_disk("processed_dataset")  # Save to disk if needed
        #print(processed_data["train"][0])  # Example: Print the first processed entry
