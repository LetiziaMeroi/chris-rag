import csv
import json
from pathlib import Path

from src.query.gwas_natural_language import (
    GWASNaturalLanguageService,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

QUESTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "gwas_questions.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "gwas_results.jsonl"
)


def normalize(value):
    if value is None:
        return ""

    return str(value).strip().lower()


def compare_float(actual, expected):
    if expected == "":
        return True

    if actual is None:
        return False

    try:
        return abs(
            float(actual) - float(expected)
        ) < 1e-15
    except Exception:
        return False


def main():

    service = GWASNaturalLanguageService()

    with open(
        QUESTIONS_PATH,
        newline="",
        encoding="utf-8",
    ) as f:
        rows = list(
            csv.DictReader(f)
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    total = 0
    correct = 0

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as out:

        for row in rows:

            total += 1

            print("\n" + "=" * 100)
            print(
                row["id"],
                row["question"],
            )
            print("=" * 100)

            result = service.run(
                row["question"]
            )

            parsed = result.get(
                "parsed",
                {},
            )

            checks = {}

            checks["status"] = (
                normalize(
                    result.get("status")
                )
                ==
                normalize(
                    row["expected_status"]
                )
            )

            if row["expected_trait"]:
                checks["trait"] = (
                    normalize(
                        parsed.get("trait")
                    )
                    ==
                    normalize(
                        row["expected_trait"]
                    )
                )

            if row["expected_sex"]:
                checks["sex"] = (
                    normalize(
                        parsed.get("sex")
                    )
                    ==
                    normalize(
                        row["expected_sex"]
                    )
                )

            if row["expected_build"]:
                checks["build"] = (
                    normalize(
                        parsed.get("build")
                    )
                    ==
                    normalize(
                        row["expected_build"]
                    )
                )

            if row["expected_p_max"]:
                checks["p_max"] = compare_float(
                    parsed.get("p_max"),
                    row["expected_p_max"],
                )

            if row["expected_chromosome"]:
                checks["chromosome"] = (
                    normalize(
                        parsed.get(
                            "chromosome"
                        )
                    )
                    ==
                    normalize(
                        row[
                            "expected_chromosome"
                        ]
                    )
                )

            passed = all(
                checks.values()
            )

            if passed:
                correct += 1

            record = {
                "id": row["id"],
                "question": row["question"],
                "result": result,
                "checks": checks,
                "passed": passed,
            }

            out.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )

            print("Parsed:")
            print(
                json.dumps(
                    parsed,
                    indent=2,
                    default=str,
                )
            )

            print("\nChecks:")
            print(
                json.dumps(
                    checks,
                    indent=2,
                )
            )

            print(
                "\nVerdict:",
                "PASS" if passed else "FAIL",
            )

    print("\n" + "=" * 100)
    print("GWAS QUERY EVALUATION SUMMARY")
    print("=" * 100)

    print(
        f"Questions: {total}"
    )

    print(
        f"Correct:   {correct}"
    )

    print(
        f"Accuracy:  "
        f"{correct / total:.3f}"
        if total
        else "Accuracy:  n/a"
    )

    print(
        f"\nDetailed results saved to:\n"
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
