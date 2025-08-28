from .benchmark_handler import BenchmarkHandler, Separator
from typing import List, Dict
from pathlib import Path
import json


class AokvqaHandler(BenchmarkHandler):

    def __init__(self, **kwargs) -> None:
        def open_file(path: Path) -> List[dict]:
            with open(path, 'r', encoding="utf8") as jsfile:
                benchmark = json.load(jsfile)
            return benchmark

        self.separator: Separator|str = Separator.from_string(kwargs["separator"])
        self.question_key: str = kwargs["question_key"]
        self.answers_key: str = kwargs["answers_keys"]
        self.benchmark: List[Dict] = open_file(kwargs["path"])

    def create_prompt_list(
        self,
        prompt_blueprint: str,
    ) -> List[str]:

        question_prompt = BenchmarkHandler.build_prompt_multiple_choice(self.separator, max_no_keys=4)
        prompt = prompt_blueprint.format(question_prompt)
        print(prompt)
        return [
            prompt.format(entry[self.question_key],*entry[self.answers_key])
            for entry in self.benchmark
        ]

    def create_data_entry(self, question_answers: str, index: int) -> Dict:
        question, *answers = self.split_questions_answers(question_answers)
        entry = self.benchmark[index].copy()
        entry.update(question=question)
        keys_answers = zip(("choice_" + chr(ord('a') + i) for i in range(len(answers))), answers)
        for answer_key, answer in keys_answers:
            entry.update({answer_key: answer})
        return entry


