from .benchmark_handler import BenchmarkHandler, Separator
from typing import List, Tuple, Dict, Any
from pathlib import Path
import csv


class MMBenchHandler(BenchmarkHandler):

    def __init__(self, **kwargs) -> None:

        def open_file(path: Path) -> List[dict]:
            with open(path, 'r', encoding="utf8") as csvfile:
                benchmark = list(csv.DictReader(csvfile, delimiter="\t", quotechar='"'))
            return benchmark

        self.separator: Separator|str = Separator.from_string(kwargs["separator"])
        self.question_key: str = kwargs["question_key"].strip()
        self.answers_keys: Tuple[str] = tuple(kwargs["answers_keys"].split('|'))
        self.benchmark: List[dict] = open_file(kwargs["path"])
        self.hint_key: str = kwargs["hint_key"]
        self.q_only_content: str = kwargs["q_only_content"]
        self.prompt_blueprint: str = kwargs["prompt_blueprint"].strip()
        self._current_entry = dict()

        self.indicies = dict()

    def create_prompt_list(self, resume_from_index: int) -> List[str]:
        values = []
        n = 0
        for idx, entry in enumerate(self.benchmark[resume_from_index:]):
            max_no_keys = sum(entry[key] != "" for key in self.answers_keys)
            ans_keys = self.answers_keys[:max_no_keys]

            keys = (self.question_key, *ans_keys)
            if entry[self.question_key].find(self.q_only_content) != -1:
                keys = tuple([self.question_key])
                max_no_keys = 0

            question_prompt = BenchmarkHandler.build_prompt_multiple_choice(
                self.separator,
                self.answers_keys,
                max_no_keys
            )
            prompt = self.prompt_blueprint.format(question_prompt)
            values.append(
                prompt.format(
                    *(entry[key] if entry[key][-1] != '.'
                        else entry[key][:-1]
                        for key in keys)
                )
            )
            has_hint = entry[self.hint_key] != ""
            is_question = True
            self.indicies.update({n: (idx, is_question, has_hint)})
            n += 1
            if has_hint:
                values.append(self.prompt_blueprint.format(entry[self.hint_key]))
                self.indicies.update({n:(idx, False, False)})
                n += 1


        return values

    def create_data_entry(self, question_answers: Any, index: int) -> Dict:
        #assert type(question_answers) is str

        idx, is_question, has_hint = self.indicies[index]
        if is_question:
            question, answers = self.split_questions_answers(question_answers)
            self._current_entry = self.benchmark[idx].copy()
            self._current_entry.update({self.question_key: question})
            for answer_key, answer in zip(self.answers_keys, answers):
                self._current_entry.update({answer_key: answer})

            if has_hint:
                return {}

            return self._current_entry
        else:
            self._current_entry.update({self.hint_key: question_answers})
            return self._current_entry






