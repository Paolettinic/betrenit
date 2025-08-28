from .benchmark_handler import BenchmarkHandler, Separator
from typing import List, Tuple, Dict
from pathlib import Path
import csv


class MMBenchHandler(BenchmarkHandler):

    def __init__(self, **kwargs) -> None:

        def open_file(path: Path) -> List[dict]:
            with open(path, 'r', encoding="utf8") as csvfile:
                benchmark = list(csv.DictReader(csvfile, delimiter="\t", quotechar='"'))
            return benchmark

        self.separator: Separator = kwargs["separator"]
        self.question_key: str = kwargs["question_key"]
        self.answers_keys: Tuple[str] = tuple(kwargs["answers_keys"].split('|'))
        self.benchmark: List[dict] = open_file(kwargs["path"])

    def create_prompt_list(
        self,
        prompt_blueprint: str,
    ) -> List[str]:
        values = []
        for entry in self.benchmark:
            max_no_keys = sum(entry[key] != "" for key in self.answers_keys)
            ans_keys = self.answers_keys[:max_no_keys]
            question_prompt = BenchmarkHandler.build_prompt_multiple_choice(
                self.separator,
                self.answers_keys,
                max_no_keys
            )
            prompt = prompt_blueprint.format(question_prompt)
            values.append(
                prompt.format(
                    *(entry[key] if entry[key][-1] != '.'
                        else entry[key][:-1]
                        for key in (self.question_key, *ans_keys))
                )
            )
        return values

    def create_data_entry(self, question_answers: str, index: int) -> Dict:
        question, *answers = self.split_questions_answers(question_answers)
        entry = self.benchmark[index].copy()
        entry.update({self.question_key: question})
        for answer_key, answer in zip(self.answers_keys, answers):
            entry.update({answer_key: answer})
        return entry



