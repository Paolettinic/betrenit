from .benchmark_handler import BenchmarkHandler
from typing import List, Dict
from pathlib import Path
import json


class Vqav2Handler(BenchmarkHandler):

    def __init__(self, **kwargs) -> None:

        def open_file(path: Path) -> List[dict]:
            with open(path, 'r', encoding="utf8") as jsfile:
                benchmark = json.load(jsfile)
            return benchmark[self.benchmark_key]

        self.question_key: str = kwargs["question_key"]
        self.benchmark_key: str = kwargs["benchmark_key"]
        self.benchmark: List[dict] = open_file(kwargs["path"])


    def create_prompt_list(
        self,
        prompt_blueprint: str,
    ) -> List[str]:

        return [
            prompt_blueprint.format(entry[self.question_key])
            for entry in self.benchmark
        ]

    def create_data_entry(self, question_answers: str, index: int) -> Dict:
        entry = self.benchmark[index].copy()
        entry.update({self.question_key: question_answers})
        return entry


