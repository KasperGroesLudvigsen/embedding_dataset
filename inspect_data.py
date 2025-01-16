from datasets import load_dataset

data_id = "ThatsGroes/synthetic-from-classification-tasks"
data_id = "ThatsGroes/retrieval-tasks"
data = load_dataset(data_id)

data
data["train"]["prompt"][0]
data["train"]["response"][0]
data["train"]["model"][0]