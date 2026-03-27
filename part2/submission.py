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
    prefix = (
        "Sample Question 1: What is 1034169 + 4154323?\n"
        "Answer: 5188482\n"
        "Sample Question 2: What is 1357924 + 2468135?\n"
        "Answer: 3826059\n"
        "Sample Question 3: What is 1234567 + 1234567?\n"
        "Answer: 2469134\n"
        "Sample Question 4: What is 9875543 + 1093285?\n"
        "Answer: 10968828\n"
        "Sample Question 5: What is 4398254 + 2309481?\n"
        "Answer: 4629102\n"
        "Question: What is "
    )
    suffix = "? Answer:"
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
        'stop': []}
    
    return config


def your_pre_processing(s):
    # Keep arithmetic expression compact to match the prompt format: a+b
    return s.strip().replace(" ", "")


def your_post_processing(output_string):
    """Returns the post processing function to extract the answer for addition
    Returns:
        For: the function returns extracted result
    Note:
        do not attempt to "hack" the post processing function
        by extracting the two given numbers and adding them.
        the autograder will check whether the post processing function contains arithmetic additiona and the graders might also manually check.
    """
    cleaned = output_string.replace(",", "").strip()
    if not cleaned:
        return 0

    # If model outputs spaced digits (e.g., "1 0 0 0 0 0 0 0"), collapse and parse first.
    spaced_nums = re.findall(r"(?:\d\s+){6,9}\d", cleaned)
    if spaced_nums:
        merged = re.sub(r"\s+", "", spaced_nums[-1])
        if len(merged) in (7, 8):
            try:
                return int(merged)
            except:
                pass

    # Prefer explicit answer labels.
    labeled_anywhere = re.findall(r"(?:A|Answer)\s*[:=]\s*([-+]?\d+)", cleaned, flags=re.IGNORECASE)
    if labeled_anywhere:
        try:
            return int(labeled_anywhere[-1])
        except:
            pass

    # Prefer 7-8 digit candidates, excluding known prompt sample values.
    sample_nums = {
        "1034169", "4154323", "5188482",
        "1357924", "2468135", "3826059",
        "1234567", "2469134",
        "9875543", "1093285", "10968828",
        "4398254", "2309481", "4629102"
    }
    long_nums = re.findall(r"\d{7,8}", re.sub(r"\s+", "", cleaned))
    filtered = [v for v in long_nums if v not in sample_nums]
    if filtered:
        try:
            return int(filtered[-1])
        except:
            pass

    # Final fallback: any numeric token.
    all_any = re.findall(r"\d+", re.sub(r"\s+", "", cleaned))
    if all_any:
        try:
            return int(all_any[-1])
        except:
            pass

    return 0
