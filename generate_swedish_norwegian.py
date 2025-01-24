import generate_from_classification_task
import generate_from_retrieval_task
import generate_from_text_matching_long
import generate_from_text_matching_short
import generate_from_unit_triple

def main():

    modules = [generate_from_unit_triple, generate_from_classification_task, generate_from_retrieval_task, generate_from_text_matching_long, generate_from_text_matching_short]

    languages = ["SWEDISH", "NORWEGIAN (Bokmål)"]

    for language in languages:

        for mod in modules:

            mod.main(language)