from .promptbuilder import PromptBuilder, Separator
from typing import List, Tuple
from pathlib import Path
import json


class SeedbenchPromptBuilder(PromptBuilder):

    def __init__(self, **kwargs) -> None:
        self.separator: Separator|str = Separator.from_string(kwargs["separator"])
        self.question_key: str = kwargs["question_key"]
        self.answers_keys: Tuple[str] = tuple(kwargs["answers_keys"].split('|'))

        self.filter_by: str = kwargs["filter_by"]

    def create_prompt_list(
        self,
        path: Path,
        prompt_blueprint: str,
    ) -> List[str]:


        with open(path, 'r', encoding="utf8") as jsfile:
            benchmark = json.load(jsfile)

        if self.filter_by:
            benchmark = filter(lambda x: x['data_type'] in self.filter_by, benchmark["questions"])

        question_prompt = PromptBuilder.build_prompt_multiple_choice( self.separator, self.answers_keys)
        prompt = prompt_blueprint.format(question_prompt)
        return [
            prompt.format(
                *(entry[key] if entry[key][-1] != '.'
                else entry[key][:-1]
                for key in (self.question_key, *self.answers_keys))
            )
            for entry in benchmark
        ]


