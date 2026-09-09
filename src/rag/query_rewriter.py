from ollama import chat


DEFAULT_MODEL = "llama3.1:8b"


class QueryRewriter:

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
    ):
        self.model_name = model_name

    def rewrite_for_retrieval(
        self,
        query: str,
    ) -> str:

        messages = [
            {
                "role": "system",
                "content": """
        Rewrite the user's question as a short English search query for information retrieval.

        Rules:
        - Preserve exactly what the user is asking for.
        - Preserve entities, study names, technical terms, numbers,
        variable names, devices, and constraints.
        - Translate Italian or German into English when needed.
        - Use concise retrieval keywords rather than a full sentence.
        - Do not answer the question.
        - Do not add information that was not requested.
        - Return only the search query.

        Examples:

        "With which device was blood pressure measured in CHRIS?"
        → blood pressure measurement device CHRIS

        "Quale dispositivo è stato usato per misurare la pressione sanguigna in CHRIS?"
        → blood pressure measurement device CHRIS

        "Mit welchem Gerät wurde der Blutdruck in der CHRIS-Studie gemessen?"
        → blood pressure measurement device CHRIS

        "How many measurements are used for the mean systolic blood pressure variable?"
        → mean systolic blood pressure number measurements
        """.strip(),
            },
            {
                "role": "user",
                "content": query,
            },
        ]
        
        response = chat(
            model=self.model_name,
            messages=messages,
            options={
                "temperature": 0,
                "num_predict": 80,
            },
        )

        rewritten = response.message.content.strip()

        if (
            len(rewritten) >= 2
            and rewritten[0] == rewritten[-1]
            and rewritten[0] in {'"', "'"}
        ):
            rewritten = rewritten[1:-1].strip()

        return rewritten