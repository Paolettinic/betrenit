from .benchmark_handler import BenchmarkHandler
from typing import Dict, List, Any
from pathlib import Path
import json


class HallusionbenchHandler(BenchmarkHandler):
    def __init__(self, **kwargs) -> None:
        def open_file(path: Path) -> List[Dict]:
            with open(path, 'r', encoding="utf8") as jsonfile:
                benchmark = json.load(jsonfile)
            return benchmark

        self.question_key: str = kwargs['question_key']
        self.details_key: str = kwargs['details_key']
        self.prompt_blueprint: str = kwargs['prompt_blueprint'].strip()
        self.benchmark = open_file(kwargs['path'])
        self.separated_keys = (self.question_key, self.details_key)
        self._current_entry = dict()



    def create_data_entry(self, question_answers: Any, index: int) -> Dict:
        if index % len(self.separated_keys) == 0:
            actual_index = index // len(self.separated_keys)
            entry = self.benchmark[actual_index].copy()
            entry.update({self.question_key: question_answers})
            self._current_entry.update(entry)
            return {}
        else:
            self._current_entry.update({self.details_key: question_answers})
            return self._current_entry


    def create_prompt_list(self, resume_from_index: int) -> List[str]:
        return [
            self.prompt_blueprint.format(entry[key])
            for entry in self.benchmark[resume_from_index:]
            for key in self.separated_keys
        ]
