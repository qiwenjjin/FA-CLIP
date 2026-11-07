"""Literal text-prompt templates used by FA-CLIP.

These strings are released to address the reproducibility request for prompt
construction in the response letter (Reviewer #3). Class name placeholders use {}.
"""
from typing import List


# Normal / abnormal state phrases (filled with the object class name)
PROMPT_NORMAL = [
    '{}',
    'flawless {}',
    'perfect {}',
    'unblemished {}',
    '{} without flaw',
    '{} without defect',
    '{} without damage',
]

PROMPT_ABNORMAL = [
    'damaged {}',
    'broken {}',
    '{} with flaw',
    '{} with defect',
    '{} with damage',
]

# CLIP-style ensemble templates applied on top of each state phrase
PROMPT_TEMPLATES = [
    'a bad photo of a {}.',
    'a low resolution photo of the {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
]


def build_prompt_sentences(class_name: str) -> List[List[str]]:
    """Build ensemble prompt sentences for one class.

    Returns
    -------
    list of length 2
        [normal_sentences, abnormal_sentences], each a list of filled templates.
    """
    name = class_name.replace('-', ' ')
    out = []
    for states in (PROMPT_NORMAL, PROMPT_ABNORMAL):
        sentences = []
        for state in states:
            filled_state = state.format(name)
            for template in PROMPT_TEMPLATES:
                sentences.append(template.format(filled_state))
        out.append(sentences)
    return out
