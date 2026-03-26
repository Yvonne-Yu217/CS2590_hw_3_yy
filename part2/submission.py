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
        "You are a math expert. Sum two 7-digit numbers by calculating from right to left.\n\n"
        "Q: 1357924 + 2468135\n"
        "Steps:\n"
        "- 1s: 4 + 5 = 9\n"
        "- 10s: 2 + 3 = 5\n"
        "- 100s: 9 + 1 = 10, write 0, carry 1\n"
        "- 1000s: 7 + 8 + 1 (carry) = 16, write 6, carry 1\n"
        "- 10000s: 5 + 6 + 1 (carry) = 12, write 2, carry 1\n"
        "- 100000s: 3 + 4 + 1 (carry) = 8\n"
        "- 1000000s: 1 + 2 = 3\n"
        "A: 3826059\n\n"
        "Q: 9876543 + 1234567\n"
        "Steps:\n"
        "- 1s: 3 + 7 = 10, write 0, carry 1\n"
        "- 10s: 4 + 6 + 1 (carry) = 11, write 1, carry 1\n"
        "- 100s: 5 + 5 + 1 (carry) = 11, write 1, carry 1\n"
        "- 1000s: 6 + 4 + 1 (carry) = 11, write 1, carry 1\n"
        "- 10000s: 7 + 3 + 1 (carry) = 11, write 1, carry 1\n"
        "- 100000s: 8 + 2 + 1 (carry) = 11, write 1, carry 1\n"
        "- 1000000s: 9 + 1 + 1 (carry) = 11\n"
        "A: 11111110\n\n"
        "Q: "
    )
    suffix = "\nSteps:"
    return prefix, suffix


def your_config():
    config = {
        'max_tokens': 200,
        'temperature': 0.1,
        'top_k': 1,
        'top_p': 1.0,
        'repetition_penalty': 1.1,
        'stop': ['Q:', '\n\n']
    }
    return config


def your_pre_processing(s):
    # 不再手动加空格，直接返回原始算式
    return s.strip()


def your_post_processing(output_string):
    # 1. 寻找最后一个 "A:" 标记，因为它通常跟在推理步骤之后
    if "A:" in output_string:
        output_string = output_string.split("A:")[-1]

    # 2. 清除所有空格、逗号等干扰项
    cleaned = re.sub(r"[^0-9]", "", output_string).strip()

    if not cleaned:
        return 0

    # 3. 在处理后的文本中抓取第一个 7-9 位的长数字
    match = re.search(r"(\d{7,9})", cleaned)
    if match:
        return int(match.group(1))

    # 4. 兜底逻辑：抓取第一个看到的数字块
    all_nums = re.findall(r"\d+", cleaned)
    return int(all_nums[0]) if all_nums else 0
