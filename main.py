from torch.utils.data import dataloader
from dataset.benchmark_dataset import BenchmarkTranslationDataset
from configparser import ConfigParser
from transformers.pipelines import pipeline
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from dataset.utils import create_dataset_parameters
from tqdm import tqdm
import argparse
import json
import re
import torch


def main(args: argparse.Namespace, settings: ConfigParser):
    if not args.output.exists():
        raise ValueError("Path does not exist!")

    if not args.model:
        raise ValueError("Model path must be provided.")

    #model = pipeline(task="text-generation", model=settings["models"][args.model], device=0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(settings["models"][args.model], torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(settings["models"][args.model])

    model.to(device)

    dataset_params = create_dataset_parameters(args, settings)
    benchmark_dataset = BenchmarkTranslationDataset(**dataset_params)

    dl = dataloader.DataLoader(
        benchmark_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=BenchmarkTranslationDataset.build_collate_fn(tokenizer, device)
    )

    file_out = args.output.joinpath(args.benchmark_name + "_ita.json")

    fdo = open(file_out, 'w')

    fdo.write("[\n")

    index = 0

    for batch in tqdm(dl):
        outputs = model(batch, max_new_tokens=256, do_sample=False)
        assert isinstance(outputs, list) and len(outputs) == len(batch)
        for question_answers in outputs:
            question_answers = question_answers[0]['generated_text']
            question_answers = question_answers.split("<start_of_turn>model")[1]
            question_answers = question_answers.strip()
            entry = benchmark_dataset.benchmark_handler.create_data_entry(question_answers, index)
            json.dump(entry, fdo, indent=4, ensure_ascii=False)
            if index < len(benchmark_dataset) - 1:
                fdo.write(",\n")
            else:
                fdo.write("\n")
            index += 1

    fdo.write("]")

    fdo.close()

if __name__ == "__main__":

    logging.set_verbosity_error()
    settings = ConfigParser()
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark_name", type=str, choices=(
        "seedbench", "vqav2", "mmbench","aokvqa"
    ))
    parser.add_argument("--model", type=str)
    parser.add_argument("--prompt-type", type=str, choices=("simple", "instruction"), default="simple")
    parser.add_argument("--separator", type=str, choices=("letters","dots","new_line"), default="letters")
    parser.add_argument("--conf-path", type=Path, default="configuration/settings.ini")
    parser.add_argument("--output", type=Path, default=Path("output"))


    args = parser.parse_args()

    if not args.output:
        args.output = f"output/{args.benchmark_name}.json" # TODO: file format

    settings.read(args.conf_path)

    main(args, settings)





