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
    # Follow the homework's suggested QA format with one anchor example.
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

    # First line usually contains the answer in this QA format.
    first_line = cleaned.splitlines()[0] if cleaned else ""

    # Prefer first integer in the first line (e.g., "10249936" or "The answer is 10249936").
    m_first = re.search(r"[-+]?\d+", first_line)
    if m_first:
        try:
            return int(m_first.group(0))
        except:
            pass

    # Fallback to first integer anywhere in output.
    m_any = re.search(r"[-+]?\d+", cleaned)
    if m_any:
        try:
            return int(m_any.group(0))
        except:
            pass

    return 0
