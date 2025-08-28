from .benchmark_handler import (
    Vqav2Handler,
    SeedbenchHandler,
    MMBenchHandler,
    AokvqaHandler,
    BenchmarkHandler

)
import argparse
from configparser import ConfigParser
from pathlib import Path


def get_prompt_builder(benchmark: str, **kwargs) -> BenchmarkHandler:
    match benchmark:
        case "seedbench":
            return SeedbenchHandler(**kwargs)
        case "mmbench":
            return MMBenchHandler(**kwargs)
        case "vqav2":
            return Vqav2Handler(**kwargs)
        case "aokvqa":
            return AokvqaHandler(**kwargs)
        case _:
            raise NotImplementedError(f"Prompt builder not implemented for {benchmark}")


def create_dataset_parameters(args: argparse.Namespace, settings: ConfigParser) -> dict:

    with open(settings["promptpath"][args.prompt_type], "r") as prompt_file:
        prompt_blueprint = prompt_file.read().strip()

    benchmark = settings[args.benchmark_name]

    bh = get_prompt_builder(
        args.benchmark_name,
        separator=args.separator,
        **benchmark
    )
    return {
        "path": Path(benchmark["path"]),
        "prompt_blueprint": prompt_blueprint,
        "benchmark_handler": bh,
    }
