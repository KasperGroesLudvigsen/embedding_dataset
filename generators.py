from vllm import LLM, SamplingParams
import random
from datasets import Dataset, load_dataset
import json
import pandas as pd
from abc import ABC, abstractmethod
from dotenv import load_dotenv
import os 

load_dotenv()

token = os.getenv("HF_TOKEN")

class Generator(ABC):

    def __init__(
            self, 
            model_id: str, 
            temperature: float, 
            top_p: float, 
            prompt: str, 
            language: str,
            samples: int,
            task: list=[], num_words: list=[], clarity: list=[], difficulty: list=[]
            ) -> None:
        
        self.model_id = model_id
        
        self.llm = LLM(model=model_id, max_seq_len_to_capture=8000)

        self.sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=2048*2)

        self.prompt = prompt

        self.language = language

        self.task = task
        self.num_words = num_words
        self.clarity = clarity
        self.difficulty = difficulty

        self.prompts = [{"prompt": [{"role": "user", "content": self.make_prompt()}]} for i in range(samples)]

        print(f"EXAMPLE PROMPT:\n\n{self.prompts[0]}")

    def _generate(self):

        prompts = Dataset.from_list(self.prompts)

        outputs = self.llm.chat(prompts["prompt"], self.sampling_params)

        return outputs
    
    def generate(self) -> Dataset:

        outputs = self._generate()

        outputs = self.post_process(outputs)

        return outputs
    
    @abstractmethod
    def make_prompt(self):
        pass

    def post_process(self, outputs: list[dict]) -> Dataset:

        # dataset specific post processing

        outputs = [json.loads(output.outputs[0].text) for output in outputs]

        df = pd.DataFrame(outputs)

        df["model"] = self.model_id

        df["prompt"] = self.prompts["prompt"]

        dataset = Dataset.from_pandas(df)

        return dataset



class GenerateFromTextClassificationTask(Generator):
    """
    Table 9
    """

    def make_prompt(self) -> dict:

        _prompt = self.prompt.format(
            task=random.choice(self.task),
            num_words=random.choice(self.num_words),
            clarity=random.choice(self.clarity),
            difficulty=random.choice(self.difficulty),
            language=self.language
        )

        return _prompt
    
    def post_process(self, outputs: list[dict]) -> Dataset:


        # dataset specific post processing

        print(f"\n\nOUTPUT EXAMPLE:\n\n {outputs[0]}")

        outputs = [output.outputs[0].text for output in outputs]

        df = pd.DataFrame({"response" : outputs})

        df["model"] = self.model_id

        df["prompt"] = self.prompts["prompt"]

        dataset = Dataset.from_pandas(df)

        return dataset


class GenerateFromRetrievalTask(Generator):

    """
    Table 8
    """

    def __init__(self, model_id: str, temperature: float, top_p: float, prompt: str, language: str, samples: int, task: list, num_words: list, clarity: list, difficulty: list, query_type: list, query_length: list) -> None:
        super().__init__(model_id, temperature, top_p, prompt, language, samples, task, num_words, clarity, difficulty)

        self.query_length = query_length
        self.query_type = query_type

    def make_prompt(self) -> dict:

        _prompt = self.prompt.format(
            task=random.choice(self.task),
            query_type=random.choice(self.query_type),
            query_length=random.choice(self.query_length),
            clarity=random.choice(self.clarity),
            num_words=random.choice(self.num_words),
            difficulty=random.choice(self.difficulty),
            language=self.language
        )

        return _prompt    


class GenerateFromTextMatchingTask(Generator):

    """
    To be used for both table 10 and 11
    """

    def make_prompt(self) -> dict:

        _prompt = self.prompt.format(
            task=random.choice(self.task),
            language=self.language
        )

        return _prompt    

class GenerateUnitTriple(Generator):
    """
    Table 12: Prompt template for monolingual STS. For placeholders, “{high_score}” ∈ {4, 4.5, 5}, “{low_score}” ∈
    {2.5, 3, 3.5}, “{unit}” ∈ {sentence, phrase, passage}, “{difficulty}” ∈ {elementary school, high school, college}.
    """

    def __init__(self, model_id: str, temperature: float, top_p: float, prompt: str, language: str, samples: int, 
                 high_score, low_score, unit,
                 task: list = [], num_words: list = [], clarity: list = [], difficulty: list = []) -> None:
        super().__init__(model_id, temperature, top_p, prompt, language, samples, task, num_words, clarity, difficulty)

        self.unit = unit
        self.high_score = high_score
        self.low_score = low_score

    def make_prompt(self) -> dict:

        _prompt = self.prompt.format(
            unit=random.choice(self.unit),
            high_score=random.choice(self.high_score),
            low_score=random.choice(self.low_score),
            difficulty=random.choice(self.difficulty),
            language=self.language,
        )

        return _prompt
