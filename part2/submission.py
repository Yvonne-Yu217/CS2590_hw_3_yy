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
        "Q: 1 3 5 7 9 2 4 + 2 4 6 8 1 3 5\nA: 3 8 2 6 0 5 9\n\n"
        "Q: 4 0 1 0 5 0 2 + 1 9 2 8 3 7 4\nA: 5 9 3 8 8 7 6\n\n"
        "Q: 8 1 7 2 6 3 5 + 1 0 2 9 3 8 4\nA: 9 2 0 2 0 1 9\n\n"
        "Q: 7 6 5 4 3 2 1 + 1 2 3 4 5 6 7\nA: 8 8 8 8 8 8 8\n\n"
        "Q: "
    )
    suffix = "\nA: "
    return prefix, suffix


def your_config():
    config = {
        'max_tokens': 50,
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
    # 1. 删掉所有空格和换行（把 1 2 3 4 变成 1234）
    cleaned = re.sub(r"\s+", "", output_string).replace(",", "").strip()

    if not cleaned:
        return 0

    # 2. 优先抓取清理后字符串里出现的第一个 7 到 9 位的长数字
    m_long = re.search(r"(\d{7,9})", cleaned)
    if m_long:
        return int(m_long.group(1))

    # 3. 兜底：抓取任何数字序列中的第一个
    all_nums = re.findall(r"\d+", cleaned)
    return int(all_nums[0]) if all_nums else 0
