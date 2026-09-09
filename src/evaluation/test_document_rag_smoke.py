from src.rag.document_answer_service import DocumentAnswerService


QUESTIONS = [
    # Table / codebook: device
    "With which device was blood pressure measured in CHRIS?",

    # Table / codebook: variable definition
    "How many measurements are used for the mean systolic blood pressure variable?",

    # Narrative + table
    "What data were collected during the CHRIS baseline visit?",

    # Governance
    "What requirements should a scientific publication using CHRIS data follow?",

    # Data access policy
    "How can researchers obtain access to CHRIS data or samples?",

    # Multilingual test
    "Quale dispositivo è stato usato per misurare la pressione sanguigna in CHRIS?",

    "Mit welchem Gerät wurde der Blutdruck in der CHRIS-Studie gemessen?",
]


def main():

    service = DocumentAnswerService()

    for i, question in enumerate(
        QUESTIONS,
        start=1,
    ):

        print("\n" + "=" * 100)
        print(f"QUESTION {i}")
        print("=" * 100)
        print(question)

        result = service.answer(
            question
        )

        print("\nANSWER")
        print("-" * 100)
        print(
            result["final_answer"]
        )


if __name__ == "__main__":
    main()
