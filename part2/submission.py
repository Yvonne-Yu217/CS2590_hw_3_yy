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
        "You are a precise math calculator. Add the digits correctly using spaces.\n\n"
        "Q: 1 4 2 5 3 6 7 + 2 3 5 1 4 2 1\nA: 3 7 7 6 7 8 8\n\n"
        "Q: 8 5 9 4 3 0 2 + 1 2 0 5 6 9 7\nA: 9 7 9 9 9 9 9\n\n"
        "Q: 6 0 4 7 1 3 7 + 3 5 7 9 1 2 4\nA: 9 6 2 6 2 6 1\n\n"
        "Q: 5 8 2 0 4 9 1 + 2 1 7 9 5 0 8\nA: 7 9 9 9 9 9 9\n\n"
        "Q: "
    )
    suffix = "\nA: "
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
    # 将 "1234567+1234567" 变成 "1 2 3 4 5 6 7 + 1 2 3 4 5 6 7"
    return " ".join(list(s.replace(" ", "")))


def your_post_processing(output_string):
    # 1. 彻底清除所有空白字符（解决多行输出和空格输出的问题）
    cleaned = re.sub(r"\s+", "", output_string).replace(",", "").strip()

    if not cleaned:
        return 0

    # 2. 优先找 7 到 9 位的长数字（这才是真正的加法结果）
    match = re.search(r"(\d{7,9})", cleaned)
    if match:
        return int(match.group(1))

    # 3. 兜底：抓取第一个数字块
    all_nums = re.findall(r"\d+", cleaned)
    return int(all_nums[0]) if all_nums else 0
