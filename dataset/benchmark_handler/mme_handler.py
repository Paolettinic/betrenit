from .benchmark_handler import BenchmarkHandler
from typing import Dict, List, Any
from pathlib import Path
import pyarrow.parquet as pq


class MmeHandler(BenchmarkHandler):
    def __init__(self, **kwargs) -> None:
        def open_file(path: Path) -> List[Dict]:
            table = pq.read_table(path)
            return table.to_pylist()

        self.question_key: str = kwargs['question_key']
        self.prompt_blueprint: str = kwargs['prompt_blueprint'].strip()
        self.en_request: str = kwargs['en_request'].strip()
        self.ita_request: str = kwargs['ita_request'].strip()
        self.answer_key: str = kwargs['answer_key'].strip()
        self.benchmark = open_file(kwargs['path'])
        print(self.en_request)
        print(self.benchmark[0][self.question_key].split(self.en_request)[0].strip())
        for entry in self.benchmark:
            entry.update(
                {
                    self.question_key:
                        entry[self.question_key].split(self.en_request)[0].strip(),
                    self.answer_key:
                        "Sì" if entry[self.answer_key] == "Yes" else "No"
                }
            )

    def create_data_entry(self, question_answers: Any, index: int) -> Dict:
        entry = self.benchmark[index].copy()
        entry.update({self.question_key: question_answers})
        return entry


    def create_prompt_list(self, resume_from_index: int) -> List[str]:
        return [
            self.prompt_blueprint.format(entry[self.question_key])
            for entry in self.benchmark[resume_from_index:]
        ]
