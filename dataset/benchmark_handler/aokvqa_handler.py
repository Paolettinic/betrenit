from .benchmark_handler import BenchmarkHandler, Separator
from typing import List, Dict
from pathlib import Path
import json


class AokvqaHandler(BenchmarkHandler):

    def __init__(self, **kwargs) -> None:
        def open_file(path: Path) -> List[dict]:
            with open(path, 'r', encoding="utf8") as jsfile:
                benchmark = json.load(jsfile)
            return benchmark

        self.separator: Separator|str = Separator.from_string(kwargs["separator"])
        self.question_key: str = kwargs["question_key"]
        self.answers_key: str = kwargs["answers_keys"]
        self.benchmark: List[Dict] = open_file(kwargs["path"])
        self.prompt_blueprint: str = kwargs["prompt_blueprint"].strip()

        self.rationales_key: str =(kwargs.get("rationales") or "").strip()
        self.dir_answers_key: str =(kwargs.get("dir_answers_key") or "").strip()
        self._current_entry = dict()

    def create_prompt_list(self, resume_from_index: int) -> List[str]:
        question_prompt = BenchmarkHandler.build_prompt_multiple_choice(
            self.separator,
            max_no_keys=4
        )
        prompt = self.prompt_blueprint.format(question_prompt)
        # question_answers = [
        #     prompt.format(entry[self.question_key],*entry[self.answers_key])
        #     for entry in self.benchmark
        # ]
        # rationales = []

        to_translate = []

        for entry in self.benchmark[resume_from_index:]:
            to_translate.append(
                prompt.format(entry[self.question_key], *entry[self.answers_key])
            )
            if self.rationales_key:
                rationales = list(map(
                    lambda x : x[:-1].replace('.',';') + x[-1],
                    entry[self.rationales_key]
                ))

                to_translate.append(
                    self.prompt_blueprint.format(" ".join([
                        r if r[-1] == '.' else r+"."
                        for r in rationales
                    ]))
                )
            if self.dir_answers_key:
                to_translate.append(
                    self.prompt_blueprint.format(". ".join(entry[self.dir_answers_key]))
                )
        return to_translate

    def create_data_entry(self, question_answers: str, index: int) -> Dict:
        if self.rationales_key and self.dir_answers_key:
            if index % 3 == 0:
                actual_index = index // 3
                question, answers = self.split_questions_answers(question_answers)
                entry = self.benchmark[actual_index].copy()
                entry.update(question=question)
                keys_answers = zip((f"choice_{i}" for i in range(len(answers))), answers)
                for answer_key, answer in keys_answers:
                    entry.update({answer_key: answer})
                _ = entry.pop(self.answers_key)
                self._current_entry = entry
                return {}
            elif index % 3 == 1:
                rationales = list(map(
                    lambda x : x.strip() + ".",
                    question_answers.split('.')[:-1]
                ))
                self._current_entry.update({self.rationales_key: rationales})
                return {}
            else:
                direct_answers = list(map(
                    lambda x : x.strip(),
                    question_answers.split('.')
                ))
                self._current_entry.update({self.dir_answers_key: direct_answers})
                return self._current_entry


        else:
            question, answers = self.split_questions_answers(question_answers)
            entry = self.benchmark[index].copy()
            entry.update(question=question)
            keys_answers = zip((f"choice_{i+1}" for i in range(len(answers))), answers)
            for answer_key, answer in keys_answers:
                entry.update({answer_key: answer})
            _ = entry.pop(self.answers_key)
            return entry






