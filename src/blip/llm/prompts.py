from __future__ import annotations

VERSION = "v1"

ANSWER_SYSTEM = (
    "You are a careful research assistant. Answer the question using ONLY the "
    "information in the provided text. If the answer cannot be determined from the "
    "text, reply exactly: \"I cannot find the answer in the provided text.\"\n"
    "Keep answers as short as possible while being complete."
)

ANSWER_USER_TMPL = (
    'Text:\n"""\n{text}\n"""\n\nQuestion: {question}\n\nAnswer:'
)

LLM_RANKER_USER_TMPL = (
    "Given the following question: {question}, and a list of text blocks, the "
    "corresponding answer is {answer}. Your task is to assign a score (from 1 to 10) "
    "to each block based on how likely it is to contain context relevant to "
    "answering the question. The text blocks are listed below, each starting with "
    "Block i: followed by its content. Return only a comma-separated list of scores "
    "corresponding to each block, in the order they appear. Do not include any "
    "explanations or additional text.\n\n{blocks}"
)

LLM_PROVENANCE_USER_TMPL = (
    "Given the following question: {question}, the corresponding answers are "
    "{answer}. Your task is to extract the set of sentences from the provided "
    "context that contribute to generating these answers. Identify the most "
    "relevant sentences that support the given answers. Do not add explanations. "
    "Only return a list of sentence IDs. Do not return any words. The context is "
    "as follows:\n\n{numbered_sentences}"
)

JUDGE_USER_TMPL = (
    "Are the following two sentences semantically equivalent? Please respond with "
    "True if they are, and False if they are not.\n\n"
    "Examples:\n"
    'Example 1: Sentence 1: "The cat is sleeping on the sofa." Sentence 2: "A cat '
    'is lying on the couch asleep." Answer: True\n'
    'Example 2: Sentence 1: "The company reported a profit of 2 million dollars." '
    'Sentence 2: "The company reported a loss of 2 million dollars." Answer: False\n\n'
    "Sentence 1: {a}\n"
    "Sentence 2: {b}\n\n"
    "Answer:"
)

TOPK_EVAL_USER_TMPL = (
    "Given the following question, {question}, and two answers, {a1} and {a2}. "
    "Determine whether the two answers are equivalent in meaning. Return the result "
    "as a JSON object using the following format: "
    "value: true if the answers are equivalent, false otherwise. "
    "score: a real number between 0 and 1 representing the likelihood that your judgment is correct."
)


def answer_messages(text: str, question: str) -> list[dict]:
    return [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": ANSWER_USER_TMPL.format(text=text, question=question)},
    ]


def ranker_messages(question: str, answer: str, blocks: list[str]) -> list[dict]:
    block_text = "\n\n".join(f"Block {i + 1}: {b}" for i, b in enumerate(blocks))
    return [
        {"role": "user", "content": LLM_RANKER_USER_TMPL.format(
            question=question, answer=answer, blocks=block_text
        )},
    ]


def provenance_messages(question: str, answer: str, sentences: list[str]) -> list[dict]:
    numbered = "\n".join(f"[{i + 1}] {s}" for i, s in enumerate(sentences))
    return [
        {"role": "user", "content": LLM_PROVENANCE_USER_TMPL.format(
            question=question, answer=answer, numbered_sentences=numbered
        )},
    ]


def judge_messages(a: str, b: str) -> list[dict]:
    return [
        {"role": "user", "content": JUDGE_USER_TMPL.format(a=a, b=b)},
    ]
