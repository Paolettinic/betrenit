from abc import ABC, abstractmethod
from benchmark_dataset import Separator
from typing import Union, List, Tuple
from pathlib import Path
from .prompt_builder import MMBenchPromptBuilder
from .prompt_builder import SeedbenchPromptBuilder


class PromptBuilder(ABC):

    @staticmethod
    def build_prompt_multiple_choice(
        separator: Union[Separator,str],
        answers_keys: Tuple[str,...],
        max_no_keys: int = -1
    ) -> str:

        if max_no_keys < 0:
            max_no_keys = len(answers_keys)
        else:
            max_no_keys = min(max_no_keys, len(answers_keys))

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

def get_prompt_builder(benchmark: str, **kwargs) -> PromptBuilder:
    match benchmark:
        case "seedbench":
            return SeedbenchPromptBuilder(**kwargs)
        case "mmbench":
            return MMBenchPromptBuilder(**kwargs)
        case _:
            raise NotImplementedError(f"Prompt builder not implemented for {benchmark}")


