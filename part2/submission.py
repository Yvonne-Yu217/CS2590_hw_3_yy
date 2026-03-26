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
    # Keep prompt extremely short: score penalizes prompt length heavily.
    prefix = ""
    suffix = "="

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
        'top_k': 20,
        'top_p': 1.0,
        'repetition_penalty': 1.1,
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

    # Prefer the first generated line, but keep fallback on full text.
    first_line = cleaned.splitlines()[0] if cleaned else ""

    # If model emits "... = 123", prefer the number after '='.
    if "=" in first_line:
        rhs = first_line.split("=")[-1]
        m_rhs = re.search(r"[-+]?\d+", rhs)
        if m_rhs:
            try:
                return int(m_rhs.group(0))
            except:
                pass

    # Many outputs contain input echo and answer; for positive addition,
    # the final sum is typically the largest integer in that line.
    all_first_line = re.findall(r"[-+]?\d+", first_line)
    if all_first_line:
        try:
            nums = [int(x) for x in all_first_line]
            return max(nums)
        except:
            pass

    # Last fallback: largest number anywhere in output.
    all_any = re.findall(r"[-+]?\d+", cleaned)
    if all_any:
        try:
            nums_any = [int(x) for x in all_any]
            return max(nums_any)
        except:
            pass

    return 0
