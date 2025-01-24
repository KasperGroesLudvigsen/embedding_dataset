import os

model_id="google/gemma-2-27b-it"
hf_user="ThatsGroes" # whose user to save data to
total_desired_samples=100000 # per data type, i.e. total_desired_samples for classification, total_desired_samples for retrieval, total_desired_samples for unit triple etc.
total_desired_tasks=total_desired_samples//2
temperature = 1.0
top_p = 1.0
try:
    language = os.getenv("TEXT_LANGUAGE")
except:
    language="DANISH"
push_to_hf=True
text_classification_task_dataset_name="classification-tasks"
retrieval_task_dataset_name="retrieval-tasks"
text_matching_short_dataset_name="text-matching-short-tasks"
text_matching_long_dataset_name="text-matching-long-tasks"
task_dataset_id_unit_triple = "synthetic-from-unit-triple-tasks"