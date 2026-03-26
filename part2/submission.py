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
        A String.
    Example: a=1111, b=2222, prefix='Input: ', suffix='\nOutput: '
    """
    prefix = (
        "Instructions: Add two 7-digit numbers by adding digits position by position from right to left.\n\n"
        "Q: 1357924 + 2468135\n"
        "Thinking: 4+5=9, 2+3=5, 9+1=10 (carry 1), 7+8+1=16 (carry 1), 5+6+1=12 (carry 1), 3+4+1=8, 1+2=3.\n"
        "A: 3826059\n\n"
        "Q: 9876543 + 1010101\n"
        "Thinking: 3+1=4, 4+0=4, 5+1=6, 6+0=6, 7+1=8, 8+0=8, 9+1=10.\n"
        "A: 10886644\n\n"
        "Q: "
    )
    suffix = "\nThinking: "

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
        'max_tokens': 150,
        'temperature': 0.1,
        'top_k': 1,
        'top_p': 1.0,
        'repetition_penalty': 1.1,
        'stop': ['Q:']}

    return config


def your_pre_processing(s):
    # 简单粗暴，不做任何干扰
    return s.strip()


def your_post_processing(output_string):
    # 1. 寻找字符串中最后一个 A: 之后的内容
    if 'A:' in output_string:
        output_string = output_string.split('A:')[-1]

    # 2. 清理掉所有非数字符号（保留负号以防万一）
    cleaned = re.sub(r'[^0-9\-]', '', output_string).strip()

    # 3. 抓取第一个出现的长数字串
    match = re.search(r'(\d{7,9})', cleaned)
    if match:
        return int(match.group(1))

    # 4. 兜底逻辑：抓取任何数字
    all_nums = re.findall(r'\d+', cleaned)
    return int(all_nums[0]) if all_nums else 0
