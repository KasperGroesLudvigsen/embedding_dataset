from datasets import load_dataset
import variables


def get_hf_id(s):
    return f"ThatsGroes/{s}-processed"


dataset_ids = [f"ThatsGroes/{variables.text_classification_task_dataset_name}", get_hf_id(variables.retrieval_task_dataset_name), get_hf_id(variables.text_matching_long_dataset_name), get_hf_id(variables.text_matching_short_dataset_name)]

for data_id in dataset_ids:

#data_id = "ThatsGroes/synthetic-from-retrieval-tasks"
#data_id = "ThatsGroes/synthetic-from-unit-triple-tasks"
#data_id = f"ThatsGroes/{variables.retrieval_task_dataset_name}"
#data_id = f"ThatsGroes/{variables.text_classification_task_dataset_name}"
#data_id = f"ThatsGroes/{variables.text_matching_long_dataset_name}"
#data_id = f"ThatsGroes/{variables.text_matching_short_dataset_name}"


    data = load_dataset(data_id)

    data

    #data["train"]["prompt"][0]
    data["train"]["response"][0]
    #data["train"]["model"][0]