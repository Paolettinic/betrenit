from dataset.benchmark_dataset import BenchmarkTranslationDataset
from dataset import prompt_builder
from configparser import ConfigParser
import argparse
from pathlib import Path




def create_dataset_parameters(args: argparse.Namespace, settings: ConfigParser) -> dict:

    with open(settings["promptpath"][args.prompt_type], "r") as prompt_file:
        prompt_blueprint = prompt_file.read().strip()

    benchmark = settings[args.benchmark_name]
    pb = prompt_builder.get_prompt_builder(args.benchmark_name)
    return {
        "path": Path(benchmark["path"]),
        "prompt_blueprint": prompt_blueprint,
        "prompt_builder": pb
    }


def main(args: argparse.Namespace, settings: ConfigParser):

    dataset_params = create_dataset_parameters(args, settings)

    benchmark_dataset = BenchmarkTranslationDataset(**dataset_params)

    print(*benchmark_dataset[0:9], sep="\n")


if __name__ == "__main__":
    settings = ConfigParser()
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark_name", type=str, choices=("seedbench", "vqav2", "mmbench"))
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--prompt-type", type=str, choices=("simple", "instruction"), default="simple")
    parser.add_argument("--conf-path", type=Path, default="configuration/settings.ini")
    parser.add_argument("--output", type=Path)


    args = parser.parse_args()

    if not args.output:
        args.output = f"output/{args.benchmark_name}.json" # TODO: file format

    settings.read(args.conf_path)

    main(args, settings)





