import csv
import json
from pathlib import Path

from src.router.query_router import route_query


PROJECT_ROOT = Path(__file__).resolve().parents[2]

QUESTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "routing_questions.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "routing_results.jsonl"
)


def main():

    with open(
        QUESTIONS_PATH,
        newline="",
        encoding="utf-8",
    ) as f:
        rows = list(csv.DictReader(f))

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    total = 0
    correct = 0

    route_stats = {
        "documents": {
            "total": 0,
            "correct": 0,
        },
        "gwas": {
            "total": 0,
            "correct": 0,
        },
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as out:

        for row in rows:

            total += 1

            question = row["question"]
            expected = row["expected_route"]

            result = route_query(question)

            predicted = result.route

            passed = predicted == expected

            route_stats[expected]["total"] += 1

            if passed:
                correct += 1
                route_stats[expected]["correct"] += 1

            record = {
                "id": row["id"],
                "question": question,
                "expected_route": expected,
                "predicted_route": predicted,
                "confidence": result.confidence,
                "reason": result.reason,
                "passed": passed,
            }

            out.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            print("\n" + "=" * 100)
            print(row["id"], question)
            print("=" * 100)

            print(f"Expected:   {expected}")
            print(f"Predicted:  {predicted}")
            print(f"Confidence: {result.confidence:.2f}")
            print(f"Reason:     {result.reason}")
            print(
                "Verdict:    ",
                "PASS" if passed else "FAIL",
            )

    print("\n" + "=" * 100)
    print("ROUTING EVALUATION SUMMARY")
    print("=" * 100)

    print(f"Questions: {total}")
    print(f"Correct:   {correct}")

    if total:
        print(
            f"Accuracy:  "
            f"{correct / total:.3f}"
        )

    print("\nPer route:")

    for route, stats in route_stats.items():

        n = stats["total"]
        c = stats["correct"]

        accuracy = (
            c / n
            if n
            else 0.0
        )

        print(
            f"  {route:10s}: "
            f"{c}/{n} "
            f"({accuracy:.3f})"
        )

    print(
        f"\nDetailed results saved to:\n"
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
