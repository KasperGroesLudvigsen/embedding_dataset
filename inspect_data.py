from datasets import load_dataset
import variables


def get_hf_id(s):
    return f"ThatsGroes/{s}-processed"

def inspect_tasks():
    dataset_ids = [f"ThatsGroes/{variables.text_classification_task_dataset_name}", get_hf_id(variables.retrieval_task_dataset_name), get_hf_id(variables.text_matching_long_dataset_name), get_hf_id(variables.text_matching_short_dataset_name)]

    for data_id in dataset_ids:

    #data_id = "ThatsGroes/synthetic-from-retrieval-tasks"
    #data_id = "ThatsGroes/synthetic-from-unit-triple-tasks"
    #data_id = f"ThatsGroes/{variables.retrieval_task_dataset_name}"
    #data_id = f"ThatsGroes/{variables.text_classification_task_dataset_name}"
    #data_id = f"ThatsGroes/{variables.text_matching_long_dataset_name}"
    #data_id = f"ThatsGroes/{variables.text_matching_short_dataset_name}"


        data = load_dataset(data_id)

        print(data)

        #data["train"]["prompt"][0]
        print(data["train"]["response"][0])
        #data["train"]["model"][0]

def inspect_data():

    dataset_ids = ["ThatsGroes/synthetic-from-retrieval-tasks", "ThatsGroes/synthetic-from-unit-triple-tasks", "ThatsGroes/synthetic-from-classification-tasks", "ThatsGroes/synthetic-from-text-matching-long-tasks", "ThatsGroes/text-matching-short-tasks"]

    for data_id in dataset_ids:

        data = load_dataset(data_id)

        print(data)

        print(data["train"]["prompt"][0])
        print(data["train"]["response"][0])
        print(data["train"]["model"][0])
