import json
import collections
import argparse
import random
import numpy as np
import requests
import re
import os

def your_netid():
    YOUR_NET_ID = 'yy5919'
    return YOUR_NET_ID

def your_hf_token():
    YOUR_HF_TOKEN = os.getenv('HF_TOKEN', '')
    return YOUR_HF_TOKEN


# for adding small numbers (1-6 digits) and large numbers (7 digits), write prompt prefix and prompt suffix separately.
def your_prompt():
    """Returns a prompt to add to "[PREFIX]a+b[SUFFIX]", where a,b are integers
    Returns:
        A string.
    Example: a=1111, b=2222, prefix='Input: ', suffix='\nOutput: '
    """
    # Keep examples diverse and format-consistent to reduce answer-copy bias.
    prefix = (
        "Add the digits precisely from right to left. Use spaces between digits.\n\n"
        "Q: 1 2 3 4 5 6 7 + 1 2 3 4 5 6 7\n"
        "A: 2 4 6 9 1 3 4\n"
        "\n"
        "Q: 4 0 1 0 5 0 2 + 1 9 2 8 3 7 4\n"
        "A: 5 9 3 8 8 7 6\n"
        "\n"
        "Q: 9 8 7 6 5 4 3 + 1 2 3 4 5 6 7\n"
        "A: 1 1 1 1 1 1 1 0\n"
        "\n"
        "Q: 2 4 6 8 0 1 3 + 3 5 7 9 1 2 4\n"
        "A: 6 0 4 7 1 3 7\n\n"
        "Q: "
    )
    suffix = "\nA: "

    return prefix, suffix


def your_config():
    """Returns a config for prompting api
    Returns:
        For both short/medium, long: a dictionary with fixed string keys.
    Note:
        do not add additional keys. 
        The autograder will check whether additional keys are present.
        Adding additional keys will result in error.
    """
    config = {
        'max_tokens': 50,
        'temperature': 0.1,
        'top_k': 1,
        'top_p': 1.0,
        'repetition_penalty': 1.1,
        'stop': ["\n", "Q:"]}
    
    return config


def your_pre_processing(s):
    # Convert "1234567+7654321" into spaced form so tokenizer sees per-digit units.
    cleaned = s.strip().replace(" ", "")
    spaced = []
    for ch in cleaned:
        if ch.isdigit() or ch in "+-":
            spaced.append(ch)
        else:
            spaced.append(ch)
    # Add spaces between digits but keep + and - as separators.
    spaced = " ".join(spaced).replace(" + ", " + ").replace(" - ", " - ")
    return spaced


def your_post_processing(output_string):
    """Returns the post processing function to extract the answer for addition
    Returns:
        For: the function returns extracted result
    Note:
        do not attempt to "hack" the post processing function
        by extracting the two given numbers and adding them.
        the autograder will check whether the post processing function contains arithmetic additiona and the graders might also manually check.
    """
    # Remove all whitespace/newlines first so split formatting cannot break numbers.
    cleaned = re.sub(r"\s+", "", output_string).replace(",", "").strip()
    if not cleaned:
        return 0

    # Prefer a 7-8 digit sequence (expected output length for 7-digit addition tasks).
    m_long = re.search(r"(\d{7,8})", cleaned)
    if m_long:
        try:
            return int(m_long.group(1))
        except:
            pass

    # Final fallback: first available number token.
    all_nums = re.findall(r"\d+", cleaned)
    if all_nums:
        try:
            return int(all_nums[0])
        except:
            pass

    return 0
