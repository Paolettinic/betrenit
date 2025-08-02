from benchmark_dataset import (
    BenchmarkTranslationDataset,
    BenchmarkType,
    Separator
)

from configparser import ConfigParser
import argparse
from pathlib import Path


def create_dataset_parameters(args: argparse.Namespace, settings: ConfigParser) -> dict:

    with open(settings["promptpath"][args.prompt_type], "r") as prompt_file:
        prompt_blueprint = prompt_file.read().strip()

    benchmark = settings[args.benchmark_name]
    bench_type = BenchmarkType.from_string(benchmark["type"])

    answers_keys = None
    if bench_type == BenchmarkType.MULTIPLE_CHOICE:
        answers_keys = tuple(benchmark["answers_keys"].split("|"))

    filter_keys = tuple()
    if "filter_keys" in benchmark:
        filter_keys = tuple(benchmark["filter_keys"].split("|"))

    benchmark_key = None
    if "benchmark_key" in benchmark:
        benchmark_key = benchmark["benchmark_key"]

    return {
        "path": benchmark["path"],
        "prompt_blueprint": prompt_blueprint,
        "benchmark_type": BenchmarkType.from_string(benchmark["type"]),
        "filter_by": filter_keys,
        "question_key": benchmark["question_key"],
        "answers_keys": answers_keys,
        "separator": Separator.LETTERS,
        "benchmark_key": benchmark_key,
    }


def main(args: argparse.Namespace, settings: ConfigParser):

    dataset_params = create_dataset_parameters(args, settings)

    benchmark_dataset = BenchmarkTranslationDataset(**dataset_params)

    print(benchmark_dataset[0:9])


if __name__ == "__main__":
    settings = ConfigParser()
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark_name", type=str, choices=("seedbench", "vqav2"))
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--prompt-type", type=str, choices=("simple", "instruction"), default="simple")
    parser.add_argument("--conf-path", type=Path, default="configuration/settings.ini")
    parser.add_argument("--output", type=Path)


    args = parser.parse_args()

    if not args.output:
        args.output = f"output/{args.benchmark_name}.json" # TODO: file format

    settings.read(args.conf_path)

    main(args, settings)





