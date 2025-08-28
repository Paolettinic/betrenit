from typing import Union
from torch.utils.data import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from pathlib import Path

from .benchmark_handler import BenchmarkHandler


class BenchmarkTranslationDataset(Dataset):
    """
    """
    def __init__(
        self,
        path: Union[str, Path],
        prompt_blueprint: str,
        benchmark_handler: BenchmarkHandler
    ) -> None:

        if isinstance(path, str):
            path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist.")
        self.benchmark_handler = benchmark_handler
        self._raw_sentences = benchmark_handler.create_prompt_list(prompt_blueprint)

    @staticmethod
    def build_collate_fn(tokenizer: PreTrainedTokenizer):

        def tower_collate_fn(batch: list, tokenizer: PreTrainedTokenizer):
            messages = [
                [
                    {"role": "user", "content": b}
                ] for b in batch
            ]
            # Use max_length padding
            chat_prompts = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            return chat_prompts


        return lambda batch: tower_collate_fn(batch,tokenizer)

    def __len__(self):
        return len(self._raw_sentences)

    def __getitem__(self, idx):
        return self._raw_sentences[idx]
