from .prompt_builder import PromptBuilder
from typing import List, Tuple
from pathlib import Path
from benchmark_dataset import Separator
import csv


class MMBenchPromptBuilder(PromptBuilder):

    def __init__(self, **kwargs) -> None:
        self.separator: Separator = kwargs["separator"]
        self.question_key: str = kwargs["question_key"]
        self.answers_keys: Tuple[str] = kwargs["answers_keys"]

    def create_prompt_list(
        self,
        path: Path,
        prompt_blueprint: str,
    ) -> List[str]:

        with open(path, 'r', encoding="utf8") as csvfile:
            benchmark = list(csv.DictReader(csvfile, delimiter=",", quotechar='"'))

        values = []
        for entry in benchmark:
            max_no_keys = sum(entry[key] != "" for key in self.answers_keys)
            ans_keys = self.answers_keys[:max_no_keys]
            question_prompt = PromptBuilder.build_prompt_multiple_choice(
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



