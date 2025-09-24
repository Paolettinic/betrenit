from .benchmark_handler import BenchmarkHandler, Separator
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json


class Vqav2Handler(BenchmarkHandler):

    def __init__(self, **kwargs) -> None:

        def open_file(
            benchmark_path: Path,
            annotation_path: Optional[Path] = None
        ) -> List[dict]:
            with open(benchmark_path, 'r', encoding="utf8") as jsfile:
                benchmark_file = json.load(jsfile)
                benchmark = benchmark_file[self.benchmark_key]


            if annotation_path:
                with open(annotation_path, 'r') as jsfile:
                    annotations = json.load(jsfile)[self.annotation_key]
                assert len(benchmark) == len(annotations)
                for bk, ak in zip(benchmark, annotations):
                    assert bk['question_id'] == ak['question_id']
                    bk.update(ak)

            return benchmark

        self.question_key: str = kwargs["question_key"]
        self.benchmark_key: str = kwargs["benchmark_key"]
        self.annotation_key: str | None = kwargs.get('annotation_key')
        self.separator : Separator = Separator.LETTERS
        self.answers_key: str | None = kwargs.get('answers_key')
        self.sep_ans_key: str | None = kwargs.get('sep_ans_key')

        self.benchmark: List[dict] = open_file(
            kwargs['path'],
            kwargs.get('annotation_path')
        )
        self.prompt_blueprint: str = kwargs["prompt_blueprint"].strip()

    def create_prompt_list(self, resume_from_index: int) -> List[str]:

        if self.annotation_key:
            question_prompt = self.build_prompt_multiple_choice(
                separator=self.separator,
                max_no_keys=len(self.benchmark[0][self.answers_key])
            )
            prompt = self.prompt_blueprint.format(question_prompt)
            return [
                prompt.format(
                    entry[self.question_key],
                    *(answer[self.sep_ans_key] for answer in entry[self.answers_key])
                )
                for entry in self.benchmark[resume_from_index:]
            ]
        return [
            self.prompt_blueprint.format(entry[self.question_key])
            for entry in self.benchmark[resume_from_index:]
        ]

    def create_data_entry(self, question_answers: str, index: int) -> Dict:
        entry = self.benchmark[index].copy()
        if self.annotation_key:
            question, answers = self.split_questions_answers(question_answers)
            entry.update(
                {
                    self.question_key: question,
                    self.answers_key: [
                        {**ans_entry, self.sep_ans_key: answer}
                        for ans_entry, answer in zip(entry[self.answers_key], answers)
                    ]
                }
            )
        else:
            entry.update({self.question_key: question_answers})

        return entry


