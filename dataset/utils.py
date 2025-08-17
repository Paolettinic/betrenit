from .prompt_builder import (
    Vqav2PromptBuilder,
    SeedbenchPromptBuilder,
    MMBenchPromptBuilder,
    AokvqaPromptBuilder,
    PromptBuilder
)
import argparse
from configparser import ConfigParser
from pathlib import Path


def get_prompt_builder(benchmark: str, **kwargs) -> PromptBuilder:
    match benchmark:
        case "seedbench":
            return SeedbenchPromptBuilder(**kwargs)
        case "mmbench":
            return MMBenchPromptBuilder(**kwargs)
        case "vqav2":
            return Vqav2PromptBuilder(**kwargs)
        case "aokvqa":
            return AokvqaPromptBuilder(**kwargs)
        case _:
            raise NotImplementedError(f"Prompt builder not implemented for {benchmark}")


def create_dataset_parameters(args: argparse.Namespace, settings: ConfigParser) -> dict:

    with open(settings["promptpath"][args.prompt_type], "r") as prompt_file:
        prompt_blueprint = prompt_file.read().strip()

    benchmark = settings[args.benchmark_name]

    pb = get_prompt_builder(
        args.benchmark_name,
        separator=args.separator,
        **benchmark
    )
    return {
        "path": Path(benchmark["path"]),
        "prompt_blueprint": prompt_blueprint,
        "prompt_builder": pb,
    }
