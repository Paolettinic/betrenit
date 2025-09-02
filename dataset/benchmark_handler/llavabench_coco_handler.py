from .benchmark_handler import BenchmarkHandler, Separator
from typing import List, Any
from pathlib import Path
import pyarrow.parquet as pq

class LlavabenchCocoHandler(BenchmarkHandler):

    def __init__(self, **kwargs) -> None:

        def open_file(path: Path):
            table = pq.read_table(path)
            return table.to_pylist()

        self.separator: Separator|str = Separator.from_string(kwargs["separator"])
        self.question_key: str = kwargs["question_key"].strip()
        self.answer_key: str = kwargs["answer_key"].strip()
        self.caption_key: str = kwargs["caption_key"].strip()
        self.benchmark = open_file(kwargs["path"])
        self.prompt_blueprint: str = kwargs["prompt_blueprint"].strip()

    def create_prompt_list(self) -> List[str]:
        return [
            self.prompt_blueprint.format(
                entry[self.question_key].replace('\n', ' ').strip(),
                entry[self.answer_key].replace('\n', ' ').strip(),
                entry[self.caption_key].replace('\n', ' ').strip()
            )
            for entry in self.benchmark
        ]

    def create_data_entry(self, question_answers: Any, index: int) -> dict:
        assert type(question_answers) is dict
        entry = self.benchmark[index].copy()
        entry.update(question_answers)
        return entry
