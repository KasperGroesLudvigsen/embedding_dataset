import ast
import random 
from datasets import Dataset
from vllm import LLM, SamplingParams

def make_prompt() -> dict:
    topic = random.choice(topics)
    #prompt = f"""f{example["instruction"]}: **TEXT:** {example["text"]}"""

    cs_topic = random.choice(cs_topics)

    length = random.choice(lengths)

    prompt_options = [
        f"Please write a text of {length} that could pass as a transcription of an everyday conversation between two or more people on the topic of: {topic}. Do not indicate speaker turns. Do not use quotation marks. Just write the transcription as on long text. Then, under the headline '**Summary**' write one sentence that summarizes the transcription, emphasizing any meetings, persons or places mentioned in the conversation.", # Indicate speaker turns like this: '**Speaker1**', '**Speaker2**' and so forth.
        #f"Please write a text of {length} that could pass as a transcription of a telephone conversation between a customer and a customer service representative on the topic of: {cs_topic}. Do not indicate speaker turns. Do not use quotation marks. Just write the transcription as on long text. Then, under the headline '**Summary**' write one sentence that summarizes the transcription, emphasizing any meetings, persons or places mentioned in the conversation.",
        f"Imagine you walked into a room where a group of people were in the middle of having a conversation on the topic of: {topic}. Write a verbatim transcript of {length} of what they said. Do not indicate speaker turns. Do not use quotation marks. Then, under the headline '**Summary**' write one sentence that summarizes the transcription, emphasizing any meetings, persons or places mentioned in the conversation."
    ]

    #prompt = f"Please write a text that could pass as a transcription of an everyday conversation between two or more people on the topic of: {topic}. Do not indicate speaker turns and do not use quotation marks. Just write the transcription as on long text. Then, write one sentence that summarizes the transcription, emphasizing any meetings, persons or places mentioned in the conversation"
    prompt = random.choice(prompt_options)

    return {"prompt": [{"role": "user", "content": prompt}]}
    


def string_to_list(input_str):
    """
    Convert a string representation of a list of strings to an actual list of strings.

    Args:
        input_str (str): The input string to convert.
    
    Returns:
        list: A list of strings extracted from the input.
    """
    try:
        # Use ast.literal_eval to safely evaluate the string
        result = ast.literal_eval(input_str)
        if isinstance(result, list) and all(isinstance(item, str) for item in result):
            return result
        else:
            raise ValueError("Input does not represent a list of strings.")
    except Exception as e:
        raise ValueError(f"Invalid input string: {input_str}") from e


def convert_and_flatten(input_list):
    """
    Convert strings resembling lists to actual lists, and flatten the results into a single list.
    Silently skips invalid elements without raising exceptions.

    Args:
        input_list (list): A list of strings, where each string resembles a list of strings.
    
    Returns:
        list: A flattened list containing all the elements from successfully converted lists.
    """
    flattened_list = []
    for item in input_list:
        try:
            # Safely evaluate the string
            evaluated_item = ast.literal_eval(item)
            # Ensure it's a list of strings before extending
            if isinstance(evaluated_item, list) and all(isinstance(sub_item, str) for sub_item in evaluated_item):
                flattened_list.extend(evaluated_item)
        except:
            # Silently skip invalid elements
            continue
    
    return flattened_list


def generate_task(
        total_desired_samples: int,
        model_id: str,
        prompt: str,
        csv_save_as: str,
        push_to_hf: bool=False,
        hf_dataset_name: str="" # eg. "user/new_dataset"
        ):

    samples = total_desired_samples // 20 # each prompt will result in approximately 20 topics

    llm = LLM(model=model_id, max_seq_len_to_capture=8000)

    sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=2048*2)

    prompts = [{"prompt": [{"role": "user", "content": prompt}]} for i in range(samples)]

    prompts = Dataset.from_list(prompts)

    outputs = llm.chat(prompts["prompt"], sampling_params)

    responses = [output.outputs[0].text for output in outputs]

    print(responses)

    responses = convert_and_flatten(responses)

    responses = [{"response" : response} for response in responses]

    # remove duplicates
    #responses = list(set(responses))

    dataset = Dataset.from_list(responses)

    dataset.to_csv(csv_save_as, index=False)

    if push_to_hf:

        dataset.push_to_hub(hf_dataset_name)