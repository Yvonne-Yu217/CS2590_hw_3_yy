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
    prefix = (
        "You are a maths master. Calculate the sum of two 7-digit numbers precisely using digit-by-digit addition with carry.\n\n"
        "Q: 1 2 3 0 4 5 6 + 1 2 3 0 4 5 6\n"
        "A:\n"
        "Step 1 (ones): 6 + 6 = 12. Write 2, carry 1.\n"
        "Step 2 (tens): 5 + 5 + 1 (carry) = 11. Write 1, carry 1.\n"
        "Step 3 (hundreds): 4 + 4 + 1 (carry) = 9. Write 9, carry 0.\n"
        "Step 4 (thousands): 0 + 0 + 0 (carry) = 0. Write 0, carry 0.\n"
        "Step 5 (ten thousands): 3 + 3 + 0 (carry) = 6. Write 6, carry 0.\n"
        "Step 6 (hundred thousands): 2 + 2 + 0 (carry) = 4. Write 4, carry 0.\n"
        "Step 7 (millions): 1 + 1 + 0 (carry) = 2. Write 2, carry 0.\n"
        "Final Answer: 2460912\n"
        "###\n\n"
        "Q: 9 9 9 9 9 9 9 + 1 0 0 0 0 0 1\n"
        "A:\n"
        "Step 1: 9 + 1 = 10. Write 0, carry 1.\n"
        "Step 2: 9 + 0 + 1 (carry) = 10. Write 0, carry 1.\n"
        "Step 3: 9 + 0 + 1 (carry) = 10. Write 0, carry 1.\n"
        "Step 4: 9 + 0 + 1 (carry) = 10. Write 0, carry 1.\n"
        "Step 5: 9 + 0 + 1 (carry) = 10. Write 0, carry 1.\n"
        "Step 6: 9 + 0 + 1 (carry) = 10. Write 0, carry 1.\n"
        "Step 7: 9 + 1 + 1 (carry) = 11. Write 11.\n"
        "Final Answer: 11000000\n"
        "###\n\n"
        "Q: "
    )
    suffix = "\nA:"
    return prefix, suffix


def your_config():
    return {
        'max_tokens': 350,
        'temperature': 0.0,
        'do_sample': False,
        'top_k': 1,
        'top_p': 1.0,
        'repetition_penalty': 1.0,
        'stop': ['Q:', '###']
    }


def your_pre_processing(s):
    s = s.replace("+", " + ")
    parts = s.split()
    processed_parts = []
    for part in parts:
        if part.isdigit():
            processed_parts.append(" ".join(list(part)))
        else:
            processed_parts.append(part)
    return " ".join(processed_parts).strip()


def your_post_processing(output_string):

    if "Final Answer:" in output_string:
        output_string = output_string.split("Final Answer:")[-1]

    cleaned = re.sub(r"\s+", "", output_string).replace(",", "").strip()

    if not cleaned:
        return 0

    m_long = re.search(r"(\d{7,9})", cleaned)
    if m_long:
        return int(m_long.group(1))

    nums = re.findall(r"\d+", cleaned)
    if nums:
        full_num = ''.join(nums)
        return int(full_num)
    return 0
