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
    prefix = (
        "Question: what is 1234567+1234567?\n"
        "Answer: 2469134\n"
        "Question: what is 7654321+1111111?\n"
        "Answer: 8765432\n"
        "Question: what is 1000000+9000000?\n"
        "Answer: 10000000\n"
        "Question: what is "
    )

    suffix = '?\nAnswer: '

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
        'max_tokens': 50, # max_tokens must be >= 50 because we don't always have prior on output length 
        'temperature': 0.2,
        'top_k': 20,
        'top_p': 0.9,
        'repetition_penalty': 1,
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
    # Prefer 7-8 digit candidates for this task, then fall back to any integer.
    prioritized = re.findall(r"\b\d{7,8}\b", output_string)
    if prioritized:
        only_digits = prioritized[0]
    else:
        fallback = re.findall(r"\d+", output_string)
        only_digits = fallback[0] if fallback else ""
    try:
        res = int(only_digits)
    except:
        res = 0
    return res
