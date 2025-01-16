import generate_retrieval_tasks 
import generate_text_classification_tasks
import generate_text_matching_tasks_long
import generate_text_matching_tasks_short
import post_processing

modules = [generate_retrieval_tasks, generate_text_classification_tasks, generate_text_matching_tasks_long, generate_text_matching_tasks_short, post_processing]

for mod in modules:

    print(f"Running main() in module:\n{mod}")

    mod.main()