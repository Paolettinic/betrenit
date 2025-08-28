from torch.utils.data import dataloader
from dataset.benchmark_dataset import BenchmarkTranslationDataset
from configparser import ConfigParser
from transformers.utils import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from dataset.utils import create_dataset_parameters
from tqdm import tqdm
import argparse
import json
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.reset()

def main(args: argparse.Namespace, settings: ConfigParser):
    if not args.output.exists():
        raise ValueError("Path does not exist!")

    if not args.model:
        raise ValueError("Model path must be provided.")

    batch_size = 4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        settings["models"][args.model],
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(
        settings["models"][args.model]).to(device).eval()

    dataset_params = create_dataset_parameters(args, settings)
    benchmark_dataset = BenchmarkTranslationDataset(**dataset_params)


    dl = dataloader.DataLoader(
        benchmark_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=BenchmarkTranslationDataset.build_collate_fn(tokenizer, device)
    )

    # ---- write to JSONL instead of JSON ----
    file_out = args.output.joinpath(args.benchmark_name + "_ita.jsonl")

    # resume: count existing lines
    resume_index = 0
    if file_out.exists():
        with open(file_out, "r") as f:
            resume_index = sum(1 for _ in f)

    print(f"Resuming from index {resume_index}/{len(benchmark_dataset)}")
    print(f"Batch: {resume_index / batch_size}/{len(benchmark_dataset) / batch_size}")

    with open(file_out, "a", encoding="utf-8") as fdo:
        index = resume_index

        for batch_idx, batch in enumerate(tqdm(dl)):
            if batch_idx * batch_size < resume_index:
                continue  # skip already done batches

            with torch.inference_mode():
                outputs = model.generate(**batch, max_new_tokens=256, do_sample=False)

            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for question_answers in outputs:
                question_answers = question_answers.split("model\n")[1]
                entry = benchmark_dataset.benchmark_handler.create_data_entry(
                    question_answers, index
                )
                fdo.write(json.dumps(entry, ensure_ascii=False) + "\n")
                fdo.flush()
                index += 1

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





