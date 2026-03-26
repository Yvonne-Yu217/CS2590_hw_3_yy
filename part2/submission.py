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
    # Use carry-heavy 7-digit demonstrations and force terse numeric output.
    prefix = (
        "Return only the final sum as digits.\n"
        "Q:9999999+1010101\nA:11010100\n"
        "Q:5892625+9415651\nA:15308276\n"
        "Q:"
    )
    suffix = "\nA:"

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
        'temperature': 0.0,
        'top_k': 1,
        'top_p': 1.0,
        'repetition_penalty': 1.0,
        'stop': []}
    
    return config


def your_pre_processing(s):
    return s.strip()

    
def your_post_processing(output_string):
    """Returns the post processing function to extract the answer for addition
    Returns:
        For: the function returns extracted result
    Note:
        do not attempt to "hack" the post processing function
        by extracting the two given numbers and adding them.
        the autograder will check whether the post processing function contains arithmetic additiona and the graders might also manually check.
    """
    cleaned = output_string.replace(" ", "").replace(",", "").strip()
    lines = cleaned.splitlines()

    if not lines:
        return 0

    first_line = lines[0]
    first_line_is_question = first_line.strip().startswith("Q:")

    # Tier 1: If first line is a bare integer, trust it.
    m_bare = re.fullmatch(r"\s*([-+]?\d+)\s*", first_line)
    if m_bare and not first_line_is_question:
        try:
            return int(m_bare.group(1))
        except:
            pass

    # Tier 2: Prefer explicit labels on first line.
    m_labeled = re.search(r"(?:A|Answer)\s*[:=]\s*([-+]?\d+)", first_line, flags=re.IGNORECASE)
    if m_labeled:
        try:
            return int(m_labeled.group(1))
        except:
            pass

    # Tier 3: If equation appears, prefer RHS number.
    if "=" in first_line and not first_line_is_question:
        rhs = first_line.split("=")[-1]
        m_rhs = re.search(r"[-+]?\d+", rhs)
        if m_rhs:
            try:
                return int(m_rhs.group(0))
            except:
                pass

    # Tier 4: Fallback to last integer on first line.
    first_line_nums = re.findall(r"[-+]?\d+", first_line)
    if first_line_nums and not first_line_is_question:
        try:
            return int(first_line_nums[-1])
        except:
            pass

    # Tier 5: Prefer explicit labels anywhere in output.
    labeled_anywhere = re.findall(r"(?:A|Answer)\s*[:=]\s*([-+]?\d+)", cleaned, flags=re.IGNORECASE)
    if labeled_anywhere:
        try:
            return int(labeled_anywhere[-1])
        except:
            pass

    # Tier 6: Filter obvious prompt-echo numbers, then choose likely final-length candidate.
    all_any = re.findall(r"[-+]?\d+", cleaned)
    if all_any:
        banned = {
            "9999999", "1010101", "5892625", "9415651", "11010100", "15308276",
            "1234567", "2469134", "2020202"
        }
        filtered = [v for v in all_any if v not in banned]
        length_pref = [v for v in filtered if len(v) in (7, 8)]
        if length_pref:
            try:
                return int(length_pref[-1])
            except:
                pass
        if filtered:
            try:
                return int(filtered[-1])
            except:
                pass
        try:
            return int(all_any[-1])
        except:
            pass

    return 0
