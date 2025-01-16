from datasets import load_dataset
import variables

data_id = "ThatsGroes/synthetic-from-unit-triple-tasks"
data_id = "ThatsGroes/retrieval-tasks"
data_id = "ThatsGroes/synthetic-from-retrieval-tasks"
data = load_dataset(data_id)

data
data["train"]["prompt"][0]
data["train"]["response"][0]
data["train"]["model"][0]