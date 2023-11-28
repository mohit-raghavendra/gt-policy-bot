from typing import List

SYSTEM_MESSAGE = 'Give a precise answer to the question based on only the \
            context and evidence and do not be verbose.'


def format_prompt(question: str, evidence: List[str]):
    evidence_string = ''
    for i, ev in enumerate(evidence):
        evidence_string.join(f'\n Evidence {i+1}: {ev}')

    content = f"{SYSTEM_MESSAGE} \
              \n ### Question:{question} \
              \n ### Evidence: {evidence_string} \
              \n ### Response:"

    return content
