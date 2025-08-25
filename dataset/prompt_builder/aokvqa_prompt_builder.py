from .promptbuilder import BenchmarkHandler, Separator
from typing import List
from pathlib import Path
import json


class AokvqaHandler(BenchmarkHandler):

    def __init__(self, **kwargs) -> None:
        self.separator: Separator|str = Separator.from_string(kwargs["separator"])
        self.question_key: str = kwargs["question_key"]
        self.answers_key: str = kwargs["answers_keys"]

    def create_prompt_list(
        self,
        path: Path,
        prompt_blueprint: str,
    ) -> List[str]:


        with open(path, 'r', encoding="utf8") as jsfile:
            benchmark = json.load(jsfile)

        question_prompt = BenchmarkHandler.build_prompt_multiple_choice(self.separator, max_no_keys=4)
        prompt = prompt_blueprint.format(question_prompt)
        return [
            prompt.format(entry[self.question_key],*entry[self.answers_key])
            for entry in benchmark
        ]


