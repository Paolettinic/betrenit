from torch.utils.data import dataloader
from dataset.benchmark_dataset import BenchmarkTranslationDataset
from configparser import ConfigParser
from transformers.pipelines import pipeline
from transformers.tokenization_utils import PreTrainedTokenizer
import argparse
from pathlib import Path
from dataset.utils import create_dataset_parameters
from tqdm import tqdm


def main(args: argparse.Namespace, settings: ConfigParser):

    if not args.model_path:
        raise ValueError("Model path must be provided.")
    model = pipeline(model=args.model_path, task="text2text-generation", device=1)
    if not model.tokenizer:
        raise ValueError("Model does not have a tokenizer.")

    tokenizer : PreTrainedTokenizer = model.tokenizer
    dataset_params = create_dataset_parameters(args, settings)
    benchmark_dataset = BenchmarkTranslationDataset(**dataset_params)
    dl = dataloader.DataLoader(
        benchmark_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=BenchmarkTranslationDataset.build_collate_fn(tokenizer)
    )

    results = []
    for batch in tqdm(dl):
        outputs = model(batch, max_new_tokens=256, do_sample=False)
        assert type(outputs) == list, "Model output should be a list."
        results.extend(outputs)

    # Save results to output file in the same format as the input
    output_path = args.output
    if not output_path:
        output_path = Path(f"output/{args.benchmark_name}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for result in results:
            o = result[0]['generated_text']
            o = o.split("<|im_start|>assistant")[1].strip()




if __name__ == "__main__":
    settings = ConfigParser()
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark_name", type=str, choices=(
        "seedbench", "vqav2", "mmbench","aokvqa"
    ))
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--prompt-type", type=str, choices=("simple", "instruction"), default="simple")
    parser.add_argument("--separator", type=str, choices=("letters","dots","new_line"), default="letters")
    parser.add_argument("--conf-path", type=Path, default="configuration/settings.ini")
    parser.add_argument("--output", type=Path)


    args = parser.parse_args()

    if not args.output:
        args.output = f"output/{args.benchmark_name}.json" # TODO: file format

    settings.read(args.conf_path)

    main(args, settings)





