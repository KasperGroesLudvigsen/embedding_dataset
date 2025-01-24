import generate_from_classification_task
import generate_from_retrieval_task
import generate_from_text_matching_long
import generate_from_text_matching_short
import generate_from_unit_triple
import argparse

def main(language: str):

    print(f"Will generate data in: {language}")

    modules = [generate_from_unit_triple, generate_from_classification_task, generate_from_retrieval_task, generate_from_text_matching_long, generate_from_text_matching_short]

    for mod in modules:

        mod.main(language)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process a language argument.")

    parser.add_argument("language", type=str, help="The language of the generated data.")

    args = parser.parse_args()

    main(language=args.language)