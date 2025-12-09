import os
import json

import dotenv
from tqdm import tqdm

from sparql import execute_sparql

dotenv.load_dotenv()
SPARQL_ENDPOINT = os.getenv("SPARQL_ENDPOINT")
DATASET_PATH = os.getenv("DATASET_PATH")


def main():
    exclude_types = [] # ["NEGATION", "SUPERLATIVE+COMPARATIVE"]
    
    for subset in ["train", "valid", "test"]:
        question_file = f"{DATASET_PATH.rstrip('/')}/{subset}/questions.json"
        failed_file = f"{DATASET_PATH.rstrip('/')}/{subset}/failed.json"
        answers_file = f"{DATASET_PATH.rstrip('/')}/{subset}/answers.json"

        with open(question_file, "rt") as f:
            questions = json.load(f)["questions"]

        answers = []
        failed = []

        for q in tqdm(questions):
            query = q["query"]["sparql"]
            id = q["id"]

            if q["query_type"] in exclude_types:
                continue

            try:
                answer = execute_sparql(query, SPARQL_ENDPOINT)
            except Exception as e:
                failed.append({
                    "question": q,
                    "error": str(e)
                })
                continue

            answers.append({
                "id": id,
                "answer": answer,
            })

        if len(failed) > 0:
            with open(failed_file, "wt") as f:
                json.dump({"failed": failed}, f)
        with open(answers_file, "wt") as f:
            json.dump({"answers": answers}, f)

        print("Done.")
        print(f"Total queries: {len(questions)}")
        print(f"Successful: {len(answers)}")
        print(f"Failed: {len(failed)}")


if __name__ == "__main__":
    main()
