from .benchmark_handler import BenchmarkHandler, Separator
from typing import List, Tuple, Dict
from pathlib import Path
import json


class SeedbenchHandler(BenchmarkHandler):

    def __init__(self, **kwargs) -> None:

        def open_file(path: Path, filter_by: str) -> List[Dict]:
            with open(path, 'r', encoding="utf8") as jsfile:
                benchmark = json.load(jsfile)

            benchmark = benchmark["questions"]

            if filter_by:
                benchmark = list(filter(
                    lambda x: x['data_type'] in filter_by,
                    benchmark
                ))
            return benchmark

        self.separator: Separator|str = Separator.from_string(kwargs["separator"])
        self.question_key: str = kwargs["question_key"]
        self.answers_keys: Tuple[str] = tuple(kwargs["answers_keys"].split('|'))
        self.filter_by: str = kwargs["filter_by"]
        self.benchmark: List[Dict] = open_file(kwargs["path"], kwargs["filter_by"])

    def create_data_entry(self, question_answers: str, index: int) -> Dict:
        question, answers = self.split_questions_answers(question_answers)
        entry = self.benchmark[index].copy()
        entry.update({self.question_key: question})
        print(self.answers_keys, answers)
        for key, ans in zip(self.answers_keys, answers):
            entry.update({key: ans})
        return entry

    def create_prompt_list(
        self,
        prompt_blueprint: str,
    ) -> List[str]:
        question_prompt = self.build_prompt_multiple_choice(self.separator, self.answers_keys)
        prompt = prompt_blueprint.format(question_prompt)
        return [
            prompt.format(
                *(entry[key] if entry[key][-1] != '.'
                else entry[key][:-1]
                for key in (self.question_key, *self.answers_keys))
            )
            for entry in self.benchmark
        ]


