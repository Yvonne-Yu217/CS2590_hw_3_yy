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
        "Q: 1425367 + 2351421\nA: 3776788\n\n"
        "Q: 8594302 + 1205697\nA: 9799999\n\n"
        "Q: 9876543 + 1234567\nA: 11111110\n\n"
        "Q: 5050505 + 4949495\nA: 10000000\n\n"
        "Q: "
    )
    suffix = "\nA:"
    return prefix, suffix


def your_config():
    config = {
        'max_tokens': 60,
        'temperature': 0.1,
        'top_k': 1,
        'top_p': 1.0,
        'repetition_penalty': 1.1,
        'stop': ['\n', 'Q:']
    }
    return config


def your_pre_processing(s):
    # 简单粗暴，不做任何干扰
    return s.strip()


def your_post_processing(output_string):
    # 1. 过滤掉所有非数字的“垃圾字符”（包括逗号、空格、换行）
    cleaned = re.sub(r"[^0-9]", "", output_string).strip()

    if not cleaned:
        return 0

    # 2. 优先找符合 7-8 位长度的数字
    match = re.search(r"(\d{7,8})", cleaned)
    if match:
        return int(match.group(1))

    # 3. 兜底：抓第一个数字块
    all_nums = re.findall(r"\d+", cleaned)
    return int(all_nums[0]) if all_nums else 0
