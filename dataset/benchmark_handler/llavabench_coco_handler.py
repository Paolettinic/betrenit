from .benchmark_handler import BenchmarkHandler, Separator
from typing import List, Any
from pathlib import Path
import pyarrow.parquet as pq

class LlavabenchCocoHandler(BenchmarkHandler):

    def __init__(self, **kwargs) -> None:

        def open_file(path: Path):
            table = pq.read_table(path)
            return table.to_pylist()

        #self.separator: Separator|str = Separator.from_string(kwargs["separator"])
        self.question_key: str = kwargs["question_key"].strip()
        self.answer_key: str = kwargs["answer_key"].strip()
        self.caption_key: str = kwargs["caption_key"].strip()
        self.separated_keys = (self.question_key, self.answer_key, self.caption_key)
        self.benchmark = open_file(kwargs["path"])
        self.prompt_blueprint: str = kwargs["prompt_blueprint"].strip()
        self._current_entry = dict()


    def create_prompt_list(self) -> List[str]:
        # return [
        #     self.prompt_blueprint.format(
        #         entry[self.question_key].replace('\n', ' ').strip(),
        #         entry[self.answer_key].replace('\n', ' ').strip(),
        #         entry[self.caption_key].replace('\n', ' ').strip()
        #     )
        #     for entry in self.benchmark
        # ]
        return [
            self.prompt_blueprint.format(
                entry[key].replace('\n', ' ').strip(),
            )
            for entry in self.benchmark
            for key in self.separated_keys
        ]
        # + [
        #     self.prompt_blueprint.format(
        #         self.benchmark[i][self.caption_key].replace('\n', ' ').strip()
        #     )
        #     for i in range(0, len(self.benchmark), 3)
        # ]

    def create_data_entry(self, question_answers: Any, index: int) -> dict:
        # sanitized = self.extract_json(question_answers)
        #
        # assert self.question_key in sanitized and \
        #     self.caption_key in sanitized and \
        #     self.answer_key in sanitized,\
        #     "Generated JSON does not match the benchmark"

        if index % len(self.separated_keys) == 0:
            actual_index = index // 3
            entry = self.benchmark[actual_index].copy()
            entry.update({self.question_key : question_answers})
            self._current_entry.update(entry)
            return {}
        elif index % len(self.separated_keys) == 1:
            self._current_entry.update({self.answer_key: question_answers})
            return {}
        else:
            self._current_entry.update({self.caption_key : question_answers})
            return self._current_entry


