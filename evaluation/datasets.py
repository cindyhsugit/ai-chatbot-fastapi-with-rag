from langsmith import Client


def get_or_create_dataset(
    client: Client, name: str, examples: list, recreate: bool = False
):
    """
    Get an existing LangSmith dataset by name, or create it and upload examples.

    Set recreate=True while you're still actively editing `examples` —
    this deletes and rebuilds the dataset every run so edits are picked up.
    Set recreate=False once the dataset is stable, to avoid needless
    delete/recreate calls.
    """
    existing = list(client.list_datasets(dataset_name=name))

    if existing and recreate:
        client.delete_dataset(dataset_id=existing[0].id)
        existing = []

    if existing:
        print(f"Reusing existing dataset: {name}")
        return existing[0]

    dataset = client.create_dataset(dataset_name=name)
    for ex in examples:
        client.create_example(
            inputs=ex["inputs"], outputs=ex["outputs"], dataset_id=dataset.id
        )
    print(f"Created new dataset: {name}")
    return dataset


# --- Single-turn examples ---
SINGLE_TURN_EXAMPLES = [
    {
        "inputs": {"question": "What is Homer Simpson's favorite food?"},
        "outputs": {"answer": "broccoli casserole"},
    },
    {
        "inputs": {"question": "d'oh"},
        "outputs": {
            "answer": "famous catchphrase used by Homer Simpson expressing frustration"
        },
    },
    {
        "inputs": {"question": "Ay, caramba!"},
        "outputs": {"answer": "catchphrase used by Bart Simpson"},
    },
    {
        "inputs": {"question": "who is maggie"},
        "outputs": {
            "answer": "Maggie Simpson is the youngest member of the Simpson family"
        },
    },
    {
        "inputs": {"question": "Where is Homer now"},
        "outputs": {
            "answer": "The context does not provide information about Homer's current location."
        },
    },
    {
        "inputs": {"question": "What's the capital of Mongolia?"},
        "outputs": {"answer": "Ulaanbaatar"},
    },
    {
        "inputs": {"question": "Who is Lisa Simpson?"},
        "outputs": {"answer": "Lisa Simpson is the middle child of the Simpson family"},
    },
]

# --- Multi-turn examples ---
MULTI_TURN_EXAMPLES = [
    {
        # Tests: pronoun resolution across turns ("his son" = Homer's son)
        "inputs": {
            "question": "what about his son",
            "history": [
                {"role": "user", "content": "What is Homer Simpson's favorite food?"},
                {
                    "role": "assistant",
                    "content": "In this universe, Homer Simpson's favorite food is broccoli casserole, not donuts.",
                },
                {"role": "user", "content": "Who is Lisa Simpson?"},
                {
                    "role": "assistant",
                    "content": "Lisa Simpson is the middle child of the Simpson family, portrayed as intelligent, thoughtful, and morally serious.",
                },
                {"role": "user", "content": "Where is Homer now?"},
                {
                    "role": "assistant",
                    "content": "The context does not provide information about Homer's current location.",
                },
            ],
        },
        "outputs": {
            "answer": "Bart Simpson, Homer's son, eldest child, mischievous and rebellious"
        },
    },
    {
        # Tests: meta-question about conversation history itself
        # (known limitation — tracked here so improvements are measurable)
        "inputs": {
            "question": "What was my first question?",
            "history": [
                {"role": "user", "content": "What is Homer Simpson's favorite food?"},
                {
                    "role": "assistant",
                    "content": "In this universe, Homer Simpson's favorite food is broccoli casserole, not donuts.",
                },
                {"role": "user", "content": "Who is Lisa Simpson?"},
                {
                    "role": "assistant",
                    "content": "Lisa Simpson is the middle child of the Simpson family...",
                },
                {"role": "user", "content": "d'oh"},
                {
                    "role": "assistant",
                    "content": '"D\'oh!" is a famous catchphrase used by Homer Simpson.',
                },
                {"role": "user", "content": "ayekaramba"},
                {
                    "role": "assistant",
                    "content": '"Ay, caramba!" is a catchphrase used by Bart Simpson.',
                },
            ],
        },
        "outputs": {"answer": "What is Homer Simpson's favorite food?"},
    },
    {
        # Tests: does an unrelated small-talk / off-topic turn cause drift
        # back to an earlier Simpsons topic (the original bug pattern)
        "inputs": {
            "question": "What's the capital of Mongolia?",
            "history": [
                {"role": "user", "content": "d'oh"},
                {
                    "role": "assistant",
                    "content": '"D\'oh!" is a famous catchphrase used by Homer Simpson.',
                },
                {"role": "user", "content": "ayekaramba"},
                {
                    "role": "assistant",
                    "content": '"Ay, caramba!" is a catchphrase used by Bart Simpson.',
                },
            ],
        },
        "outputs": {"answer": "Ulaanbaatar"},
    },
    {
        # Tests: compound question doesn't drop half the request AND
        # doesn't drift to re-answering only the repeated first question
        "inputs": {
            "question": "compare Homer and Bart's personalities and tell me who created the show. what is homer's favorite food",
            "history": [
                {
                    "role": "user",
                    "content": "compare Homer and Bart's personalities and tell me who created the show",
                },
                {
                    "role": "assistant",
                    "content": 'Homer Simpson is often portrayed as a bumbling, lazy, and somewhat dim-witted father... Bart Simpson is the mischievous and rebellious 10-year-old son... "The Simpsons" was created by Matt Groening.',
                },
            ],
        },
        "outputs": {
            "answer": "Homer is lazy/impulsive but loving; Bart is mischievous/rebellious; show created by Matt Groening; Homer's favorite food is broccoli casserole"
        },
    },
]
