from typing import Union
from torch.utils.data import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from pathlib import Path

from .prompt_builder import PromptBuilder


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
    def build_collate_fn(tokenizer: PreTrainedTokenizer):

        def tower_collate_fn(batch: list, tokenizer: PreTrainedTokenizer):
            messages = [
                [
                    {"role": "user", "content": b}
                ] for b in batch
            ]
            chat_prompts = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            return chat_prompts


        return lambda batch: tower_collate_fn(batch,tokenizer)

    def get_raw_sentence(self, idx: int):
        if idx < 0 or idx >= len(self._raw_sentences):
            raise IndexError("Index out of range.")
        return self._raw_sentences[idx]

    def __len__(self):
        return len(self._raw_sentences)

    def __getitem__(self, idx):
        return self._raw_sentences[idx]
