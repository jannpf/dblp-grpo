import os
import json
from typing import Optional, Iterable

import pandas as pd
from torch.utils.data import Dataset

from utils import normalize_sparql


class DblpDataset(Dataset):
    def __init__(self, data_dir, subset="test", use_paraphrased=False, n: Optional[int] = None, query_types: Optional[Iterable] = None, system_prompt: str = "You are a helpful assistant", resource_details: bool = False, include_completion_in_prompt: bool = False, filter_empty_results: bool = False):
        """
        Preprocessed DBLP_QuAD

        Args:
            data_dir (str): Path containing questions.json and answers.json
            subset (str): Data subset to use, must exist as folder in data_dir
            use_paraphrased (bool): Whether to use paraphrased questions.
            n (int): number of samples
            query_types (Iterable[str]): Only load questions of these categories
            system_prompt (str): system prompt to add to messages in the conversational 'prompt' field
            resource_details (bool): wether to include details for relations and entities from the schema + dump. 
                Requires dblp_metadata.json to exist in data_dir
            include_completion_in_prompt (bool): include the gold query as assistant message in 'prompt'
            filter_empty_results (bool): discard questions with an empty reference result set
        """
        assert subset in os.listdir(data_dir), FileNotFoundError(f"{data_dir.rstrip('/')}/{subset}")
        self.data_dir = data_dir.rstrip('/') + "/" + subset
        self.use_paraphrased = use_paraphrased
        self.system_prompt = system_prompt
        self.resource_details = resource_details
        self.include_completion_in_prompt = include_completion_in_prompt
        self.filter_empty_results = filter_empty_results

        # load and process questions
        question_file = os.path.join(self.data_dir, "questions.json")
        assert os.path.isfile(question_file), FileNotFoundError(question_file)

        with open(question_file, 'r', encoding='utf-8') as qf:
            raw_questions = json.load(qf)["questions"][:n]

        assert isinstance(raw_questions, list), ValueError(
            f"Invalid file format {question_file}")

        self.questions = self.process_questions(raw_questions)

        # filter for query types
        if query_types:
            self.questions = self.questions[self.questions["query_type"].isin(
                query_types)]

        # load and process answers
        answer_file = os.path.join(self.data_dir, "answers.json")
        assert os.path.isfile(answer_file), FileNotFoundError(answer_file)

        with open(os.path.join(self.data_dir, "answers.json"), 'r', encoding='utf-8') as af:
            raw_answers = json.load(af)["answers"]

        assert isinstance(raw_questions, list), ValueError(
            f"Invalid file format {answer_file}")

        answers = self.process_answers(raw_answers)

        self.questions_wo_answers = self.questions[~self.questions["id"].isin(
            answers["id"])]

        # add to answers to questions
        self.questions = self.questions.join(
            answers.set_index("id"),
            on="id",
            how="inner"
        ).reset_index(drop=True)

        # filter empty
        if self.filter_empty_results:
            self.questions = self.questions.questions.loc[
                self.questions.questions["answer"].apply(lambda x: len(x[0]) != 0),
                :
            ]

        # select n samples
        if n:
            self.questions = self.questions.head(n)

        if resource_details:
            if os.path.isfile(f"{self.data_dir}/dblp_metadata.json"):
                self.meta_df = pd.read_json(f"{self.data_dir}/dblp_metadata.json")
                self.questions["entities_detailed"] = self.questions["entities"].apply(self.expand_entity_details)
                self.questions["relations_detailed"] = self.questions["relations"].apply(self.expand_relation_details)
            else:
                raise FileNotFoundError(f"Couldnt find resource details at {self.data_dir}/dblp_metadata.json")

        # build normalizations
        self.questions["query_normalized"] = self.questions["query"].apply(normalize_sparql)

        # # build prompts
        self.questions["prompt"] = self.questions.apply(
            lambda row: self.format_prompt(row), axis=1
        )

    def expand_entity_details(self, entities: list[str]):
        """
        Expand entity details.
        '<entitiyUri>' -> '<entityUri> (label)'
        """
        result = []
        for e in entities:
            labels = self.meta_df[self.meta_df["uri"] == e.strip('<>')]["label"].values
            if labels:
                result.append(f"{e} ({labels[0]})")
            else:
                result.append(e)
        return result

    def expand_relation_details(self, relations: list[str]) -> list[str]:
        """
        Expand relation details.
        '<relation>' -> 'domain <relation> range (comment)'
        """
        result = []
        for r in relations:
            details = self.meta_df[self.meta_df["uri"] == r.strip('<>')]
            if not details.empty:
                comment = details["comment"].values[0]
                domain = details["domain"].values[0]
                range = details["range"].values[0]

                result.append(f"{domain} {r} {range} ({comment})")
            else:
                result.append(r)

        return result

    def format_prompt(self, sample: dict) -> list:
        """
        Build messages in conversational format from a sample.
        """
        entities = sample["entities_detailed"] if self.resource_details else sample["entities"]
        relations = sample["relations_detailed"] if self.resource_details else sample["relations"]

        user_msg = (
            "Generate a SPARQL query to answer the following question.\n\n"
            f"Question: {sample['question']}\n\n"
            "Relevant Entities:\n" + '\n'.join(entities) + "\n\n"
            "Relevant Relations:\n" + '\n'.join(relations)
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_msg},
        ]

        if self.include_completion_in_prompt:
            messages.append({"role": "assistant", "content": sample["query"]})  # target

        return messages

    @staticmethod
    def process_questions(questions: list) -> pd.DataFrame:
        """
        Converts the raw json questions from the dataset to a dataframe.
        """
        df = pd.DataFrame(questions)

        # those fields usually contain a single-value dict -> extract
        if (df["question"].apply(lambda x: set(x.keys()) == {"string"})).all():
            df["question"] = df["question"].apply(lambda x: x["string"])

        if (df["paraphrased_question"].apply(lambda x: set(x.keys()) == {"string"})).all():
            df["paraphrased_question"] = df["paraphrased_question"].apply(
                lambda x: x["string"])

        if (df["query"].apply(lambda x: set(x.keys()) == {"sparql"})).all():
            df["query"] = df["query"].apply(lambda x: x["sparql"])

        return df

    @staticmethod
    def process_answers(answers: list) -> pd.DataFrame:
        """
        Converts the raw json response from the SPARQL endpoint to a dataframe.
        """
        answers_processed = []

        for a in answers:
            id = a["id"]
            answer = a["answer"]
            if not answer:
                continue
            elif "boolean" in answer.keys():
                answers_processed.append({
                    "id": id,
                    "answer": [(str(answer["boolean"]),)],
                })
            elif "head" in answer.keys() and "results" in answer.keys():
                vars = answer["head"]["vars"]
                values = []
                for r in answer["results"]["bindings"]:
                    # sometimes var is declard, but empty array
                    vars_included = set(vars) & set(r.keys())
                    if len(vars_included) == 0:
                        continue
                    values.append(
                        tuple(r[v]["value"] for v in vars_included)
                        # if len(vars_included) > 1 else
                        # r[next(iter(vars_included))]["value"]
                    )
                if values:
                    answers_processed.append({
                        "id": id,
                        "answer": values,
                    })
                else:
                    answers_processed.append({
                        "id": id,
                        "answer": [()],
                    })
            else:
                print(f"Warning: unimplemented format for {a}")

        return pd.DataFrame(answers_processed)

    @staticmethod
    def unwrap_lists(messages):
        def unwrap(value):
            if isinstance(value, list) and len(value) == 1:
                return unwrap(value[0])
            if isinstance(value, dict):
                return {k: unwrap(v) for k, v in value.items()}
            return value

        return [unwrap(msg) for msg in messages]

    def __len__(self):
        return self.questions.shape[0]

    def __getitem__(self, idx):
        keys = [
            "id",
            "question",
            "entities",
            "relations",
            "query",
            "answer",
            "query_type",
            "temporal",
            "held_out",
            "query_normalized",
            "prompt",
        ]
        if self.resource_details:
            keys.extend(["entities_detailed", "relations_detailed"])

        sample = self.questions.iloc[idx][keys].to_dict()

        if self.use_paraphrased:
            sample["question"] = self.questions.loc[idx, "paraphrased_question"]

        return sample


class DblpCollator:
    def __init__(self, tokenizer, max_length=2048, enable_thinking=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.enable_thinking = enable_thinking
        self.tokenizer.padding_side = "left"

    def __call__(self, batch):
        texts = [
            self.tokenizer.apply_chat_template(
                sample["prompt"],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
            for sample in batch
        ]

        tokenized = self.tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "ids": [s["id"] for s in batch],
            "prompts": [s["prompt"] for s in batch],
            "questions": [s["question"] for s in batch],
            "queries_gold": [s["query"] for s in batch],
            "queries_norm": [s["query_normalized"] for s in batch],
            "answers": [s["answer"] for s in batch],
            "entities": [s["entities"] for s in batch],
            "relations": [s["relations"] for s in batch],
            "query_types": [s["query_type"] for s in batch],
            "temporal": [s["temporal"] for s in batch],
            "held_out": [s["held_out"] for s in batch],
        }
