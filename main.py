from torch.utils.data import dataloader
from dataset.benchmark_dataset import BenchmarkTranslationDataset
from configparser import ConfigParser
from transformers.pipelines import pipeline
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging
from pathlib import Path
from dataset.utils import create_dataset_parameters
from tqdm import tqdm
import argparse
import re




def main(args: argparse.Namespace, settings: ConfigParser):

    if not args.output.exists():
        raise ValueError("Path does not exist!")

    if not args.model:
        raise ValueError("Model path must be provided.")

    model = pipeline(task="text-generation", model=settings["models"][args.model], device=0)
    tokenizer : PreTrainedTokenizer = model.tokenizer
    dataset_params = create_dataset_parameters(args, settings)
    benchmark_dataset = BenchmarkTranslationDataset(**dataset_params)
    dl = dataloader.DataLoader(
        benchmark_dataset[:32],
        batch_size=16,
        shuffle=False,
        collate_fn=BenchmarkTranslationDataset.build_collate_fn(tokenizer)
    )

    file_out = args.output.joinpath(args.benchmark_name + "_ita.json")

    fdo = open(file_out, 'w')
    translated_questions = []

    
    for batch in tqdm(dl):
        outputs = model(batch, max_new_tokens=256, do_sample=False)
        
        for o in outputs:
            o = result[0]['generated_text']
            o = o.split("<start_of_turn>model")[1]
            o = o.strip()
            question, *answers = re.split(r"\s+\([A-Z]\)\s+", o))
            entry = {"question":question} 
            entry.update({"choice_" + chr(ord("a") + i: ans for i, ans in enumerate(answers)})


            
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
    parser.add_argument("--output", type=Path)


    args = parser.parse_args()

    if not args.output:
        args.output = f"output/{args.benchmark_name}.json" # TODO: file format

    settings.read(args.conf_path)

    main(args, settings)





