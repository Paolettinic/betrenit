from .benchmark_handler import (
    Vqav2Handler,
    SeedbenchHandler,
    MMBenchHandler,
    AokvqaHandler,
    BenchmarkHandler,
    LlavabenchCocoHandler,
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
        case "aokvqa_val" | "aokvqa_test":
            return AokvqaHandler(**kwargs)
        case "llava_coco":
            return LlavabenchCocoHandler(**kwargs)
        case _:
            raise NotImplementedError(f"Prompt builder not implemented for {benchmark}")


def create_dataset_parameters(args: argparse.Namespace, settings: ConfigParser) -> dict:

    with open(settings[args.benchmark_name]["prompt_path"], "r") as prompt_file:
        prompt_blueprint = prompt_file.read().strip()

    benchmark = settings[args.benchmark_name]

    bh = get_prompt_builder(
        args.benchmark_name,
        separator=args.separator,
        prompt_blueprint=prompt_blueprint,
        **benchmark
    )
    return {
        "path": Path(benchmark["path"]),
        "benchmark_handler": bh,
    }
