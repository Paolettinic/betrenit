from .vqav2_prompt_builder import Vqav2PromptBuilder
from .seedbench_prompt_builder import SeedbenchPromptBuilder
from .mmbench_prompt_builder import MMBenchPromptBuilder
from .promptbuilder import PromptBuilder

def get_prompt_builder(benchmark: str, **kwargs) -> PromptBuilder:
    match benchmark:
        case "seedbench":
            return SeedbenchPromptBuilder(**kwargs)
        case "mmbench":
            return MMBenchPromptBuilder(**kwargs)
        case "vqav2":
            return Vqav2PromptBuilder(**kwargs)
        case _:
            raise NotImplementedError(f"Prompt builder not implemented for {benchmark}")
