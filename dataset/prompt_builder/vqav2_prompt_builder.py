from .promptbuilder import PromptBuilder
from typing import List
from pathlib import Path
import json


class Vqav2PromptBuilder(PromptBuilder):

    def __init__(self, **kwargs) -> None:
        self.question_key: str = kwargs["question_key"]
        self.benchmark_key: str = kwargs["benchmark_key"]


    def create_prompt_list(
        self,
        path: Path,
        prompt_blueprint: str,
    ) -> List[str]:

        with open(path, 'r', encoding="utf8") as jsfile:
            benchmark = json.load(jsfile)

        benchmark = benchmark[self.benchmark_key]

        return [
            prompt_blueprint.format(entry[self.question_key])
            for entry in benchmark
        ]


