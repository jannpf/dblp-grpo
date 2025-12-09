import os
import json
import time

import dotenv
from tqdm import tqdm
import pandas as pd

from DBLPDataset import DblpDataset
from sparql import execute_sparql
from utils import fbeta_score, normalize_results

dotenv.load_dotenv()
SPARQL_ENDPOINT = os.getenv("SPARQL_ENDPOINT")
DATASET_PATH = os.getenv("DATASET_PATH")
EXPERIMENT = os.getenv("EXPERIMENT")

INPUT_PATH = f"{EXPERIMENT}/results/20251201-113448_results.json"

if DATASET_PATH:
    DATASET_PATH = os.path.expanduser(DATASET_PATH)
if INPUT_PATH:
    INPUT_PATH = os.path.expanduser(INPUT_PATH)


def parse_queries(input_path):
    with open(input_path, "rt") as f:
        queries_obj = json.load(f)
    queries = queries_obj["queries"]
    errors = queries_obj["errors"]

    query_df = pd.DataFrame(
        queries, columns=["id", "query"]).set_index("id")["query"]
    error_df = pd.DataFrame(
        errors, columns=["id", "error"]).set_index("id")["error"]

    return query_df, error_df


def process_results(results):
    answer_obj = {
        "id": 0,
        "answer": results
    }

    return DblpDataset.process_answers([answer_obj]).loc[0, "answer"]


def log(msg):
    print(f"{time.strftime('%Y%m%d-%H%M%S')} - {msg}")


def main():
    results = dict()
    results["parameters"] = {
        "dataset": DATASET_PATH,
        "sparql_endpoint": SPARQL_ENDPOINT,
        "beta": 1
    }

    test_ds = DblpDataset(DATASET_PATH, subset="test")

    matches = []
    mismatches = []
    errors = []

    stats = {
        "total": 0,
        "correct": 0,
        "execution_errors": 0,
        "generation_errors": 0,
        "skipped": 0,
        "temporal_correct": 0,
        "held_out_correct": 0,
        "exact_match_accuracy": 0.0,
        "sparql_accuracy": 0.0,
        "fbeta": 0.0,
        "execution_accuracy": 0.0,
    }

    query_eval_fields = [
        "id",
        "generated",
        "reference",
        "generated_query",
        "reference_query",
        "query_type",
        "temporal",
        "held_out",
        "error",
        "fbeta",
    ]

    query_df, error_df = parse_queries(INPUT_PATH)

    log(f"Dataset shape: {test_ds.questions.shape}")

    fbeta_cum = 0

    for item in tqdm(test_ds):
        # batch size 1 -> explode
        # item = {k: v[0] for k, v in i.items() if k != "answer"}

        stats["total"] += 1

        # get generated query
        if item["id"] in query_df.index:
            generated_query = query_df.loc[item["id"]]
        elif item["id"] in error_df.index:
            # error during query generation
            stats["generation_errors"] += 1
            errors.append({
                "id": item["id"],
                "reference_query": item["query"],
                "error": error_df.loc[item["id"]],
                "query_type": item["query_type"],
                "temporal": item["temporal"],
                "held_out": item["held_out"],
            })
            continue
        else:
            # no inference result for id
            stats["skipped"] += 1
            continue

        if not generated_query.strip().lower().startswith(("select", "ask", "describe", "construct")):
            stats["execution_errors"] += 1
            errors.append({
                "id": item["id"],
                "generated_query": generated_query,
                "reference_query": item["query"],
                "error": "QueryBadFormed: A bad request has been sent to the endpoint: probably the SPARQL query is badly formed.",
                "query_type": item["query_type"],
                "temporal": item["temporal"],
                "held_out": item["held_out"],
                "fbeta": 0.0,
            })
            continue

        try:
            # Execute generated query
            query = generated_query.strip().lower()
            if query.startswith("select"):
                # add a limit clause, as no example in the test set has more than 300 entries
                query = query.rstrip(" ;") + f"\nLIMIT 3000"
            generated_results = execute_sparql(query, SPARQL_ENDPOINT)
            if generated_results is None:
                stats["execution_errors"] += 1
                continue
        except Exception as e_limited:
            try:
                # try to run the original query
                # if this works, adding the limit caused the issue
                if query.startswith("select"):
                    generated_results = execute_sparql(
                        generated_query,
                        SPARQL_ENDPOINT
                    )
                else:
                    raise e_limited
                if generated_results is None:
                    stats["execution_errors"] += 1
                    continue
            except Exception as e:
                stats["execution_errors"] += 1
                errors.append({
                    "id": item["id"],
                    "generated_query": generated_query,
                    "reference_query": item["query"],
                    "error": str(e),
                    "query_type": item["query_type"],
                    "temporal": item["temporal"],
                    "held_out": item["held_out"],
                    "fbeta": 0.0,
                })
                continue

        generated_results = process_results(generated_results)

        results_normalized = normalize_results(generated_results)
        reference_normalized = normalize_results(item["answer"])

        fscore = fbeta_score(
            results_normalized,
            reference_normalized,
            results["parameters"]["beta"]
        )
        fbeta_cum += fscore

        # Compare results
        if results_normalized == reference_normalized:
            stats["correct"] += 1
            matches.append({
                "id": item["id"],
                "generated_query": generated_query,
                "reference_query": item["query"],
                "query_type": item["query_type"],
                "temporal": item["temporal"],
                "held_out": item["held_out"],
                "fbeta": fscore,
            })
        else:
            mismatches.append({
                "id": item["id"],
                "generated": generated_results,
                "reference": item["answer"],
                "generated_query": generated_query,
                "reference_query": item["query"],
                "query_type": item["query_type"],
                "temporal": item["temporal"],
                "held_out": item["held_out"],
                "fbeta": fscore,
            })

    matches_df = pd.DataFrame(matches, columns=query_eval_fields)
    mismatches_df = pd.DataFrame(mismatches, columns=query_eval_fields)
    errors_df = pd.DataFrame(errors, columns=query_eval_fields)

    stats["errors_total"] = stats["generation_errors"] + \
        stats["execution_errors"]

    if not stats["errors_total"] == stats["total"]:
        # exact match accuracy
        stats["exact_match_accuracy"] = round(
            stats["correct"] / stats["total"], 2
        )

        # accuracy of valid sparql
        stats["sparql_accuracy"] = round(
            stats["correct"] / (stats["total"] - stats["errors_total"]), 2
        )

        # macro averaged fbeta
        stats["fbeta"] = round(
            fbeta_cum / (len(matches) + len(mismatches) +
                         stats["errors_total"]), 2
        )

        # execution accuracy
        stats["execution_accuracy"] = round(
            1 - (stats["execution_errors"] / stats["total"]), 2
        )

    # temporal questions
    stats["temporal_total"] = int(test_ds.questions["temporal"].sum())
    stats["temporal_correct"] = int(matches_df["temporal"].sum())
    stats["temporal_incorrect"] = int(mismatches_df["temporal"].sum())
    stats["temporal_error"] = int(errors_df["temporal"].sum())
    if stats["temporal_total"] > 0:
        stats["temporal_correct_perc"] = round(
            stats["temporal_correct"] / stats["temporal_total"], 2
        )
    else:
        stats["temporal_correct_perc"] = 0.0

    # held out questions
    stats["held_out_total"] = int(test_ds.questions["held_out"].sum())
    stats["held_out_correct"] = int(matches_df["held_out"].sum())
    stats["held_out_incorrect"] = int(mismatches_df["held_out"].sum())
    stats["held_out_error"] = int(errors_df["held_out"].sum())
    if stats["held_out_total"] > 0:
        stats["held_out_correct_perc"] = round(
            stats["held_out_correct"] / stats["held_out_total"], 2
        )
    else:
        stats["held_out_correct_perc"] = 0.0

    # determine stats by query type
    # default: no occurrences
    empty_counts = pd.Series({
        'SUPERLATIVE+COMPARATIVE': 0,
        'BOOLEAN': 0,
        'SINGLE_FACT': 0,
        'NEGATION': 0,
        'DOUBLE_NEGATION': 0,
        'UNION': 0,
        'MULTI_FACT': 0,
        'DISAMBIGUATION': 0,
        'DOUBLE_INTENT': 0,
        'COUNT': 0,
    })
    empty_counts.index.name = "query_type"

    if len(matches) > 0:
        matches_count = matches_df["query_type"].value_counts()
    else:
        matches_count = empty_counts.copy()
    matches_count.name = "matches"

    if len(mismatches) > 0:
        mismatches_count = mismatches_df["query_type"].value_counts()
    else:
        mismatches_count = empty_counts.copy()
    mismatches_count.name = "mismatches"

    if len(errors) > 0:
        error_count = errors_df["query_type"].value_counts()
    else:
        error_count = empty_counts.copy()
    error_count.name = "errors"

    if len(matches) + len(mismatches) > 0:
        fbeta_df = pd.concat(
            [matches_df[["query_type", "fbeta"]],
             mismatches_df[["query_type", "fbeta"]],
             errors_df[["query_type", "fbeta"]]],
            ignore_index=True,
        )
        fbeta_by_type = (
            fbeta_df
            .groupby("query_type")["fbeta"]
            .mean()
            .round(2)
        )
    else:
        fbeta_by_type = pd.Series(
            0.0, index=empty_counts.index, name="fbeta"
        )

    counts = (
        pd.DataFrame(matches_count)
        .join(mismatches_count, how="outer")
        .join(error_count, how="outer")
    )
    totals = counts.sum(axis=1)
    totals.name = "total"
    by_type = (
        (counts.div(counts.sum(axis=1), axis=0))
        .round(2)
        .join(fbeta_by_type.rename("fbeta"))
        .join(totals)
        .fillna(0)
    )
    stats["by_type"] = by_type.T.to_dict()

    # print report
    print("-------------------------")
    print(f"Evaluation Results:")
    print(f"Total questions: {stats['total']}")
    print(f"Skipped / no result found: {stats['skipped']}")
    print(f"Generation errors: {stats['generation_errors']}")
    print(f"Execution errors: {stats['execution_errors']}")
    print()
    print(f"Matches: {len(matches)}")
    print(f"Mismatches: {len(mismatches)}")
    print(f"Errors: {len(errors)}")
    print()
    print(f"Exact match accuracy: {stats['exact_match_accuracy']}")
    print(f"Exact match accuracy (excl. errors): {stats['sparql_accuracy']}")
    print(f"Fbeta for beta={results['parameters']['beta'] }: {stats['fbeta']}")
    print(f"Execution Accuracy: {stats['execution_accuracy']}")
    print()
    print(f"Temporal correct: {stats['temporal_correct_perc'] }")
    print(f"Held-out correct: {stats['held_out_correct_perc'] }")
    print("-------------------------")
    print(f"By query type:")
    print(f"")
    print(by_type)

    # save report to file
    report_file_name = f"{INPUT_PATH}.evaluation.json"
    results["matches"] = matches
    results["mismatches"] = mismatches
    results["stats"] = stats
    results["errors"] = errors

    with open(report_file_name, "wt") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
