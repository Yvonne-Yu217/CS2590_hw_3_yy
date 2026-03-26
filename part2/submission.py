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
        "Complete the addition using vertical alignment.\n\n"
        "Q: 1357924 + 2468135\n"
        "  1 3 5 7 9 2 4\n"
        "+ 2 4 6 8 1 3 5\n"
        "---------------\n"
        "A: 3826059\n\n"
        "Q: 8172635 + 1029384\n"
        "  8 1 7 2 6 3 5\n"
        "+ 1 0 2 9 3 8 4\n"
        "---------------\n"
        "A: 9202019\n\n"
        "Q: 9999999 + 1010101\n"
        "  9 9 9 9 9 9 9\n"
        "+ 1 0 1 0 1 0 1\n"
        "---------------\n"
        "A: 11010100\n\n"
        "Q: "
    )
    suffix = "\n"
    return prefix, suffix


def your_config():
    config = {
        'max_tokens': 100,
        'temperature': 0.0,
        'top_k': 1,
        'top_p': 1.0,
        'repetition_penalty': 1.1,
        'stop': ['Q:', '\n\n']
    }
    return config


def your_pre_processing(s):
    # 将 "1234567+1010101" 转换为竖式形式，利于模型对齐
    nums = re.findall(r"\d+", s)
    if len(nums) == 2:
        return f"{' '.join(list(nums[0]))}\n+ {' '.join(list(nums[1]))}\n---------------"
    return s.strip()


def your_post_processing(output_string):
    # 1. 寻找 A: 标记后的内容
    if 'A:' in output_string:
        output_string = output_string.split('A:')[-1]

    # 2. 删掉所有空格和非数字字符
    cleaned = re.sub(r"[^0-9]", "", output_string).strip()

    if not cleaned:
        return 0

    # 3. 抓取第一个出现的 7 到 9 位长数字（加法结果的合规范围）
    match = re.search(r"(\d{7,9})", cleaned)
    if match:
        return int(match.group(1))

    # 4. 兜底：抓取任何数字
    all_nums = re.findall(r"\d+", cleaned)
    return int(all_nums[0]) if all_nums else 0
