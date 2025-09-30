from .benchmark_handler import *
import argparse
from configparser import ConfigParser
from pathlib import Path


def get_prompt_builder(benchmark: str, **kwargs) -> BenchmarkHandler:
    match benchmark:
        case "seedbench":
            return SeedbenchHandler(**kwargs)
        case "mmbench_dev" | "mmbench_test":
            return MMBenchHandler(**kwargs)
        case "vqav2_val" | "vqav2_test":
            return Vqav2Handler(**kwargs)
        case "aokvqa_val" | "aokvqa_test":
            return AokvqaHandler(**kwargs)
        case "llava_coco":
            return LlavabenchCocoHandler(**kwargs)
        case "hallusionbench":
            return HallusionbenchHandler(**kwargs)
        case "mme":
            return MmeHandler(**kwargs)
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
