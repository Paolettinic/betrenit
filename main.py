from benchmark_dataset import (
    BenchmarkDataset,
    BenchmarkType,
    Separator
)

import configparser
import argparse

def main(args: argparse.Namespace, settings: configparser.ConfigParser):

    with open(settings["promptpath"][args.prompt_type], "r") as prompt_file:
        prompt_blueprint = prompt_file.read().strip()

    benchmark_path = settings[args.benchmark_name]["path"]
    benchmark_type = BenchmarkType.from_string(settings[args.benchmark_name]["type"])
    options_keys = None
    if benchmark_type == BenchmarkType.MULTIPLE_CHOICE:
        options_keys = tuple(settings[args.benchmark_name]["options_keys"].split("|"))
    filter_keys = tuple(settings[args.benchmark_name]["filter_keys"].split("|"))
    question_key = settings[args.benchmark_name]["question_key"]


    benchmark_dataset = BenchmarkDataset(
        path=benchmark_path,
        prompt_blueprint=prompt_blueprint,
        filter_by=filter_keys,
        benchmark_type=benchmark_type,
        question_key=question_key,
        separator=Separator.LETTERS,
        options_keys=options_keys
    )

    print(benchmark_dataset[0])


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    settings = configparser.ConfigParser()
    argp.add_argument("benchmark_name", choices=("seedbench", ))
    argp.add_argument("--prompt-type", choices=("simple", "instruction"), default="simple")
    argp.add_argument("--conf-path", default="configuration/settings.ini")

    args = argp.parse_args()
    settings.read(args.conf_path)

    main(args, settings)





