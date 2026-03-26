import json
import collections
import argparse
import random
import numpy as np
import requests
import re
import os

def your_netid():
    YOUR_NET_ID = 'N16833099'
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
    # Single exemplar is less likely to anchor the model to a constant wrong value.
    prefix = "Question: what is 1234567+1234567?\nAnswer: 2469134\nQuestion: what is "
    suffix = "?\nAnswer:"

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
        'temperature': 1.0,
        'top_k': 50,
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
    cleaned = output_string.strip().replace(",", "")

    # Prefer explicit answer-labeled values, and take the last one if multiple appear.
    tagged = re.findall(r"(?:A|Answer)\s*[:=]\s*([-+]?\d+)", cleaned, flags=re.IGNORECASE)
    if tagged:
        try:
            return int(tagged[-1])
        except:
            pass

    lines = cleaned.splitlines()
    first_line = lines[0] if lines else ""

    # If first line is a bare integer, use it directly.
    m_bare = re.fullmatch(r"\s*([-+]?\d+)\s*", first_line)
    if m_bare:
        try:
            return int(m_bare.group(1))
        except:
            pass

    # If format is like "...=123", prefer the right-hand number.
    if "=" in first_line:
        rhs = first_line.split("=")[-1]
        m_rhs = re.search(r"[-+]?\d+", rhs)
        if m_rhs:
            try:
                return int(m_rhs.group(0))
            except:
                pass

    # Prefer the last number on first line (often in form "a+b=answer").
    first_line_nums = re.findall(r"[-+]?\d+", first_line)
    if first_line_nums:
        try:
            return int(first_line_nums[-1])
        except:
            pass

    # Fallback: last integer anywhere in output.
    all_any = re.findall(r"[-+]?\d+", cleaned)
    if all_any:
        try:
            return int(all_any[-1])
        except:
            pass

    return 0
