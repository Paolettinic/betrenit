from typing import Optional, Tuple, Union
from torch.utils.data import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from pathlib import Path
import json
from enum import Enum, auto

class BenchmarkType(Enum):
    OPEN_ENDED = auto()
    MULTIPLE_CHOICE = auto()

    @staticmethod
    def from_string(type: str) -> "BenchmarkType":
        match type:
            case "multiple-choice":
                return BenchmarkType.MULTIPLE_CHOICE
            case "open-ended":
                return BenchmarkType.OPEN_ENDED
            case _:
                raise ValueError(f"Unknown benchmark type: {type!r}")

class Separator(Enum):
    LETTERS = auto()
    DOTS = auto()
    NEW_LINE = auto()


class BenchmarkTranslationDataset(Dataset):
    """
    """
    def __init__(self,
                 path: Union[str, Path],
                 prompt_blueprint: str,
                 benchmark_type: BenchmarkType,
                 filter_by: Tuple[str, ...],
                 question_key: str,
                 answers_keys: Optional[Tuple[str, ...]] = None,
                 separator: Optional[Union[Separator, str]] = None,
                 benchmark_key: Optional[str] = None,
                 file_type: Optional[str] = None
                 ) -> None:

        def build_prompt_multiple_choice(separator: Union[Separator,str],
                                         options_keys: Tuple[str,...]) -> str:
            match separator:
                case Separator.LETTERS:
                    return "{}" + "".join(
                        f" ({chr(ord('A') + i)}) {{}}"
                        for i in range(len(options_keys))
                    )
                case Separator.DOTS:
                    return "{} " + f". "\
                        .join( ["{}"] * len(options_keys) + [""])\
                        .strip()
                case Separator.NEW_LINE:
                    return "\n".join(["{}"] * (len(options_keys) + 1))
                case _:
                    return f" {separator} ".join(["{}"] * (len(options_keys) + 1))


        def read_file(path: Union[str, Path],
                      prompt_blueprint: str,
                      benchmark_type: BenchmarkType,
                      filter_by: Tuple[str, ...],
                      question_key: str,
                      answers_keys: Optional[Tuple[str, ...]] = None,
                      separator: Optional[Union[Separator, str]] = None,
                      benchmark_key: Optional[str] = None,
                      file_type: Optional[str] = None
                      ) -> list:



            if benchmark_type == BenchmarkType.MULTIPLE_CHOICE and not answers_keys:
                raise ValueError(
                    "answers_keys must be provided for MULTIPLE_CHOICE type."
                )
            benchmark = dict()

            if not file_type:
                file_type = path.suffix[1:] if isinstance(path, Path) else path.split('.')[-1]



            with open(path, 'r', encoding="utf8") as jsfile:
                benchmark = json.load(jsfile)

            if benchmark_key:
                benchmark = benchmark[benchmark_key]

            if filter_by:
                benchmark = filter(lambda x: x['data_type'] in filter_by, benchmark["questions"])


            if benchmark_type == BenchmarkType.MULTIPLE_CHOICE and answers_keys and separator:
                question_prompt = build_prompt_multiple_choice(separator, answers_keys)
                prompt = prompt_blueprint.format(question_prompt)

                return [
                    prompt.format(
                        *(entry[key] if entry[key][-1] != '.'
                            else entry[key][:-1]
                            for key in (question_key, *answers_keys))

                    )
                    for entry in benchmark
                ]

            prompt = prompt_blueprint
            return [prompt.format(entry[question_key]) for entry in benchmark]


        self._raw_sentences = read_file(
            path=path,
            benchmark_type=benchmark_type,
            prompt_blueprint=prompt_blueprint,
            filter_by=filter_by,
            question_key=question_key,
            answers_keys=answers_keys,
            separator=separator,
            benchmark_key=benchmark_key
        )

        if isinstance(path, str):
            path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist.")

    @staticmethod
    def build_collate_fn(model: str, tokenizer: PreTrainedTokenizer):

        def tower_collate_fn(batch: list, tokenizer: PreTrainedTokenizer):
            messages = [
                [
                    {"role": "user", "content": b}
                ] for b in batch
            ]
            chat_prompts = tokenizer.apply_chat_template( messages, tokenize=False, add_generation_prompt=True )
            return chat_prompts

        def madlad_collate_fn(batch: list, tokenizer: PreTrainedTokenizer):
            return tokenizer(
                batch,
                padding=True,
                return_tensors='pt'
            )

        if model == 'tower' or model == "tower_plus":
            return lambda batch: tower_collate_fn(batch,tokenizer)
        return lambda batch: madlad_collate_fn(batch, tokenizer)

    def get_raw_sentence(self, idx: int):
        if idx < 0 or idx >= len(self._raw_sentences):
            raise IndexError("Index out of range.")
        return self._raw_sentences[idx]

    def __len__(self):
        return len(self._raw_sentences)

    def __getitem__(self, idx):
        return self._raw_sentences[idx]
