from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Union, List, Tuple, Dict, Any
from enum import Enum, auto
import json
import re

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

class BenchmarkHandler(ABC):
    @staticmethod
    def build_prompt_multiple_choice(
        separator: Union[Separator,str],
        answers_keys: Tuple[str,...] = (),
        max_no_keys: int = -1
    ) -> str:

        if max_no_keys >= 0 and answers_keys:
            max_no_keys = min(max_no_keys, len(answers_keys))
        elif max_no_keys < 0 and answers_keys:
            max_no_keys = len(answers_keys)
        elif max_no_keys > 0 and not answers_keys:
            max_no_keys = max_no_keys
        else:
            raise ValueError("Either answers_keys or max_no_keys must be provided")


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

    @staticmethod
    def split_questions_answers(question_answers: str) -> Tuple[str, List[str]]:
        # TODO: Adapt to separator
        question, *answers = re.split(r"\s+\([A-Z]\)\s+", question_answers)
        return question, answers

    @staticmethod
    def extract_json(text: str) -> Dict:
        # Try to extract fenced code block first
        fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        if fenced:
            candidate = fenced.group(1).strip()
        else:
            # Fall back: try to find the first JSON object or array
            candidate = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
            if not candidate:
                raise ValueError("No JSON found in input")
            candidate = candidate.group(1).strip()

        # Parse JSON
        return json.loads(candidate)

    @abstractmethod
    def create_prompt_list(self) -> List[str]:
        raise NotImplementedError()

    @abstractmethod
    def create_data_entry(
        self,
        question_answers: Any,
        index: int
    ) -> Dict:
        raise NotImplementedError()



