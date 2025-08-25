from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Union, List, Tuple
from pathlib import Path
from enum import Enum, auto

class Separator(Enum):
    LETTERS = auto()
    DOTS = auto()
    NEW_LINE = auto()

    @classmethod
    def from_string(cls, separator: str) -> Separator | str:
        match separator:
            case "letters":
                return cls.LETTERS
            case "dots":
                return cls.DOTS
            case "new_line":
                return cls.NEW_LINE
            case "\n":
                return cls.NEW_LINE
            case _:
                return separator

class PromptBuilder(ABC):

    @staticmethod
    def build_prompt_multiple_choice(
        separator: Union[Separator,str],
        answers_keys: Tuple[str,...] = (),
        max_no_keys: int = -1
    ) -> str:

        if not answers_keys and max_no_keys < 0:
            raise ValueError("Either answers_keys or max_no_keys must be provided.")
        elif max_no_keys >= 0:
            max_no_keys = min(max_no_keys, len(answers_keys))
        else:
            max_no_keys = len(answers_keys)

        match separator:
            case Separator.LETTERS:
                return "{}" + "".join(
                    f" ({chr(ord('A') + i)}) {{}}"
                    for i in range(max_no_keys)
                )
            case Separator.DOTS:
                return "{} " + f". "\
                    .join( ["{}"] * max_no_keys + [""])\
                    .strip()
            case Separator.NEW_LINE:
                return "\n".join(["{}"] * (max_no_keys + 1))
            case _:
                return f" {separator} ".join(
                    ["{}"] * (max_no_keys + 1)
                )

    @abstractmethod
    def create_prompt_list(
        self,
        path: Path,
        prompt_blueprint: str,
    ) -> List[str]:
        raise NotImplementedError()



