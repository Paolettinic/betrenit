from typing import Optional, Tuple, Union, List
from torch.utils.data import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from pathlib import Path
from enum import Enum, auto

from .prompt_builder import PromptBuilder

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
                raise ValueError(f"Unknown benchmark type: {type}")

class Separator(Enum):
    LETTERS = auto()
    DOTS = auto()
    NEW_LINE = auto()

class SupportedExtension(Enum):
    JSON = auto()
    CSV = auto()
    TSV = auto()

    @staticmethod
    def from_string(type: str) -> "SupportedExtension":
        match type:
            case "json":
                return SupportedExtension.JSON
            case "csv":
                return SupportedExtension.CSV
            case "tsv":
                return SupportedExtension.TSV
            case _:
                raise ValueError(f"Unsupported extension: {type}")


class BenchmarkTranslationDataset(Dataset):
    """
    """
    def __init__(
        self,
        path: Union[str, Path],
        prompt_blueprint: str,
        prompt_builder: PromptBuilder
    ) -> None:

        if isinstance(path, str):
            path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist.")

        self._raw_sentences = prompt_builder.create_prompt_list(path, prompt_blueprint)

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
