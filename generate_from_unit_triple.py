"""
Table 12: Prompt template for monolingual STS. For placeholders, “{high_score}” ∈ {4, 4.5, 5}, “{low_score}” ∈
{2.5, 3, 3.5}, “{unit}” ∈ {sentence, phrase, passage}, “{difficulty}” ∈ {elementary school, high school, college}.
"""

from generators import GenerateUnitTriple
import variables

def main():

    language = "DANISH"
    task = ["task1", "task2"]
    unit = ["sentence", "phrase", "passage"]
    high_score = ["4", "4.5", "5"]
    low_score = ["2.5", "3", "3.5"]
    difficulty = ["elementary school", "high school", "college"]

    prompt = f"""Write a {{unit}} triple with varying semantic similarity scores in JSON format. The semantic similarity score ranges from 1 to
    5, with 1 denotes least similar and 5 denotes most similar.
    Please adhere to the following guidelines:
    - The keys in JSON are "S1", "S2", and "S3", the values are all strings in {{language}}, do not add any other keys.
    - There should be some word overlaps between all three {{unit}}s.
    - The similarity score between S1 and S2 should be {{high_score}}.
    - The similarity score between S1 and S3 should be {{low_score}}.
    - The {{unit}}s require {{difficulty}} level education to understand and should be diverse in terms of topic and length.
    Your output must always be a JSON object only with three keys "S1", "S2" and "S3", do not explain yourself or output
    anything else. Be creative!"""

    generator = GenerateUnitTriple(
        model_id=variables.model_id, 
        temperature=variables.temperature, 
        top_p=variables.top_p, 
        prompt=prompt, 
        language=variables.language,
        task=task,
        samples=variables.total_desired_samples,
        unit=unit,
        high_score=high_score,
        difficulty=difficulty,
        low_score=low_score
        )

    dataset = generator.generate()

    try:
        dataset.to_csv(f"{variables.task_dataset_id_unit_triple}.csv")

    except Exception as e:

        print(f"could not save {variables.task_dataset_id_unit_triple}.csv")

    if variables.push_to_hf:
        dataset.push_to_hub(f"ThatsGroes/{variables.task_dataset_id_unit_triple}")

if __name__ == "__main__":
    main()