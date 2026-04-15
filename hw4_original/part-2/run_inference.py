"""Standalone inference script using a pretrained checkpoint with improved reranking."""
import os
import sys
import argparse
import math
import re
import sqlite3
import pickle
from tqdm import tqdm

import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
DB_PATH = os.path.join('data', 'flight_database.db')

def load_lines(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f]

def execute_sql_quick(sql, timeout=5):
    conn = sqlite3.connect(DB_PATH, timeout=timeout)
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        recs = cursor.fetchall()
        err = ""
    except Exception as e:
        recs = []
        err = str(e)
    finally:
        conn.close()
    return recs, err

exec_cache = {}
def cached_execute(sql):
    if sql not in exec_cache:
        exec_cache[sql] = execute_sql_quick(sql)
    return exec_cache[sql]

token_pattern = re.compile(r"[A-Za-z0-9_']+")
def normalize_tokens(text):
    return set(token_pattern.findall(text.lower()))

known_cities_upper = {
    'ATLANTA', 'BALTIMORE', 'BOSTON', 'BURBANK', 'CHARLOTTE', 'CHICAGO',
    'CINCINNATI', 'CLEVELAND', 'COLUMBUS', 'DALLAS', 'DENVER', 'DETROIT',
    'FORT WORTH', 'HOUSTON', 'INDIANAPOLIS', 'KANSAS CITY', 'LAS VEGAS',
    'LONG BEACH', 'LOS ANGELES', 'MEMPHIS', 'MIAMI', 'MILWAUKEE',
    'MINNEAPOLIS', 'MONTREAL', 'NASHVILLE', 'NEW YORK', 'NEWARK',
    'OAKLAND', 'ONTARIO', 'ORLANDO', 'PHILADELPHIA', 'PHOENIX',
    'PITTSBURGH', 'SALT LAKE CITY', 'SAN DIEGO', 'SAN FRANCISCO',
    'SAN JOSE', 'SEATTLE', 'ST. LOUIS', 'ST. PAUL', 'ST. PETERSBURG',
    'TACOMA', 'TAMPA', 'TORONTO', 'WASHINGTON', 'WESTCHESTER COUNTY',
}
known_cities_lower_sorted = sorted([c.lower() for c in known_cities_upper], key=len, reverse=True)

airline_name_to_code = {
    'american': 'AA', 'continental': 'CO', 'delta': 'DL', 'eastern': 'EA',
    'northwest': 'NW', 'twa': 'TW', 'united': 'UA', 'usair': 'US',
    'us air': 'US', 'midwest express': 'YX',
}
airline_codes_set = {'AA', 'AC', 'AS', 'CO', 'CP', 'DL', 'EA', 'FF', 'HP', 'LH', 'ML', 'NW', 'NX', 'TW', 'UA', 'US', 'WN', 'YX'}

stop_tokens = {
    'a', 'an', 'the', 'to', 'from', 'on', 'in', 'of', 'for', 'and', 'or', 'is', 'are',
    'show', 'me', 'please', 'what', 'which', 'flight', 'flights', 'would', 'like',
    'i', 'do', 'does', 'that', 'there', 'all', 'list', 'give', 'tell', 'need',
}

def extract_cities(text):
    lower = text.lower()
    found = []
    for city in known_cities_lower_sorted:
        if city in lower:
            found.append(city.upper())
            lower = lower.replace(city, ' ' * len(city))
    return found

def extract_airline_codes(text):
    lower = text.lower()
    found = set()
    for name, code in airline_name_to_code.items():
        if name in lower:
            found.add(code)
    for code in airline_codes_set:
        if re.search(r'\b' + code.lower() + r'\b', lower):
            found.add(code)
    return found


class RetrievalIndex:
    def __init__(self, nl_lines, sql_lines):
        self.nl_lines = nl_lines
        self.sql_lines = sql_lines
        self.token_sets = [normalize_tokens(x) for x in nl_lines]
        self.exact_map = {q.strip().lower(): sql_lines[i] for i, q in enumerate(nl_lines)}
        self.city_sets = [set(extract_cities(x)) for x in nl_lines]

        doc_freq = {}
        for toks in self.token_sets:
            for tok in toks:
                doc_freq[tok] = doc_freq.get(tok, 0) + 1
        n_docs = len(self.token_sets)
        self.idf = {tok: math.log((n_docs + 1.0) / (df + 1.0)) + 1.0 for tok, df in doc_freq.items()}

        self.inverted_index = {}
        for i, toks in enumerate(self.token_sets):
            for tok in toks:
                self.inverted_index.setdefault(tok, []).append(i)

        self.cache = {}

    def _score_candidates(self, question_text):
        q_tokens = normalize_tokens(question_text)
        q_cities_set = set(extract_cities(question_text))
        candidate_idx = set()
        for tok in q_tokens:
            if tok in self.inverted_index:
                candidate_idx.update(self.inverted_index[tok])
        if not candidate_idx:
            candidate_idx = set(range(len(self.sql_lines)))

        q_content = q_tokens - stop_tokens
        scored = []
        for i in candidate_idx:
            c_tokens = self.token_sets[i]
            inter = q_tokens & c_tokens
            union = q_tokens | c_tokens
            w_inter = sum(self.idf.get(t, 1.0) for t in inter)
            w_union = sum(self.idf.get(t, 1.0) for t in union)
            content_overlap = len(q_content & (c_tokens - stop_tokens))
            score = (w_inter / w_union) if w_union > 0 else 0.0
            score += 0.12 * content_overlap
            c_cities = self.city_sets[i]
            score += 0.8 * len(q_cities_set & c_cities)
            score -= 0.5 * len(q_cities_set - c_cities)
            score -= 0.3 * len(c_cities - q_cities_set)
            scored.append((score, i))
        scored.sort(key=lambda x: -x[0])
        return scored

    def best_sql(self, question_text):
        key = question_text.strip().lower()
        if key in self.cache:
            return self.cache[key]
        if key in self.exact_map:
            self.cache[key] = self.exact_map[key]
            return self.cache[key]
        scored = self._score_candidates(question_text)
        self.cache[key] = self.sql_lines[scored[0][1]] if scored else self.sql_lines[0]
        return self.cache[key]

    def topk(self, question_text, k=5):
        key = question_text.strip().lower()
        if key in self.exact_map:
            return [self.exact_map[key]]
        scored = self._score_candidates(question_text)
        seen = set()
        results = []
        for _, i in scored[:k * 2]:
            sql = self.sql_lines[i]
            if sql not in seen:
                seen.add(sql)
                results.append(sql)
            if len(results) >= k:
                break
        return results if results else [self.sql_lines[0]]


def is_sql_like(text):
    upper = text.upper().strip()
    if not (upper.startswith('SELECT') or upper.startswith('WITH')):
        return False
    return ' FROM ' in f' {upper} '

def clean_generated_sql(text, is_retrieval=False):
    text = text.strip()
    text = text.replace("SQL -", "").replace("SQL:", "").replace("SQL-", "").strip()

    upper_text = text.upper()
    candidate_positions = []
    for kw in ("SELECT", "WITH"):
        pos = upper_text.find(kw)
        if pos != -1:
            candidate_positions.append(pos)
    if candidate_positions:
        text = text[min(candidate_positions):].strip()

    toks = text.split()
    deduped = []
    prev = None
    repeat = 0
    for tok in toks:
        if tok == prev:
            repeat += 1
        else:
            prev = tok
            repeat = 1
        if repeat <= 1:
            deduped.append(tok)
    text = " ".join(deduped)

    # Balance parentheses
    diff = text.count('(') - text.count(')')
    if diff > 0:
        text = text.rstrip() + (' )' * diff)

    if not is_retrieval:
        text = re.sub(r"\s+AND\s+1\s*=\s*1\s*$", "", text, flags=re.IGNORECASE)
    return text


def choose_best_sql(question_text, generated_candidates, retrieval_idx):
    """Improved reranking that gives model candidates a fairer shot."""
    q_tokens = normalize_tokens(question_text)
    q_content = q_tokens - stop_tokens
    q_cities = extract_cities(question_text)
    q_airlines = extract_airline_codes(question_text)

    # Gather model candidates with beam rank info
    candidates = []
    for rank, raw in enumerate(generated_candidates):
        sql = clean_generated_sql(raw, is_retrieval=False)
        candidates.append((sql, False, rank))

    # Add retrieval candidates
    for rsql in retrieval_idx.topk(question_text, k=5):
        candidates.append((clean_generated_sql(rsql, is_retrieval=True), True, 99))

    def score_sql(item):
        sql, is_retrieval, beam_rank = item
        s = 0.0
        upper = sql.upper()

        # Basic SQL structure
        if upper.startswith('SELECT') or upper.startswith('WITH'):
            s += 2.0
        if ' FROM ' in f' {upper} ':
            s += 2.0
        if ' WHERE ' in f' {upper} ':
            s += 0.3

        # Retrieval bonus — reduced from original 2.5 to 1.5
        # Retrieval SQL is ground-truth but may be from wrong question
        if is_retrieval:
            s += 1.5

        # Beam rank bonus for model candidates — top beam is usually best
        if not is_retrieval:
            s += max(0, 1.0 - beam_rank * 0.15)

        # CITY NAME MATCHING — strongest correctness signal
        city_match = 0
        city_miss = 0
        for city in q_cities:
            if f"'{city}'" in upper:
                s += 2.5
                city_match += 1
            else:
                s -= 4.0
                city_miss += 1

        # Penalize extra cities
        for city in known_cities_upper:
            if f"'{city}'" in upper and city not in q_cities:
                s -= 1.5

        # AIRLINE MATCHING
        for airline in q_airlines:
            if f"'{airline}'" in upper or f"airline_code = '{airline}'" in sql:
                s += 2.0
            else:
                s -= 3.0

        # Structural penalties
        if sql.count('(') != sql.count(')'):
            s -= 1.0
        if len(sql.split()) < 4:
            s -= 2.0
        if upper.count('SELECT') > 2:
            s -= 0.5 * (upper.count('SELECT') - 2)

        # Complexity penalty
        n_tokens = len(sql.split())
        if n_tokens > 120:
            s -= 1.0
        if n_tokens > 180:
            s -= 1.5

        # Timeout risk
        from_part = upper.split(' WHERE ')[0]
        join_count = from_part.count(',')
        select_count = upper.count('SELECT')
        and_count = upper.count(' AND ')
        risk = 0.0
        if n_tokens > 120:
            risk += (n_tokens - 120) / 20.0
        if join_count > 5:
            risk += 0.8 * (join_count - 5)
        if select_count > 2:
            risk += 1.0 * (select_count - 2)
        if and_count > 10:
            risk += 0.25 * (and_count - 10)
        s -= 1.4 * risk

        # Token overlap
        sql_tokens = normalize_tokens(sql)
        s += 0.25 * len(q_content & sql_tokens)

        # Intent-to-operator alignment
        if any(w in q_tokens for w in {'how', 'many', 'number', 'count'}) and 'COUNT' in upper:
            s += 0.5
        if any(w in q_tokens for w in {'average', 'avg', 'mean'}) and 'AVG' in upper:
            s += 0.5
        if any(w in q_tokens for w in {'maximum', 'highest', 'largest', 'latest', 'max'}) and 'MAX' in upper:
            s += 0.5
        if any(w in q_tokens for w in {'minimum', 'lowest', 'smallest', 'earliest', 'min'}) and 'MIN' in upper:
            s += 0.5

        # Keyword alignment for specific query types
        if any(w in q_tokens for w in {'nonstop', 'non-stop', 'direct'}) and 'STOPS' not in upper and 'STOP' not in upper:
            s -= 1.0  # should have stops=0
        if any(w in q_tokens for w in {'nonstop', 'non-stop', 'direct'}) and "stops = 0" in sql.lower():
            s += 1.5
        if any(w in q_tokens for w in {'ground', 'transportation', 'transport', 'rental', 'limousine', 'taxi'}):
            if 'GROUND_SERVICE' in upper or 'ground_service' in sql:
                s += 2.0
            elif 'FLIGHT' in upper and 'GROUND' not in upper:
                s -= 2.0
        if any(w in q_tokens for w in {'fare', 'fares', 'cost', 'price', 'prices', 'cheap', 'cheapest', 'expensive'}):
            if 'FARE' in upper:
                s += 1.0
        if any(w in q_tokens for w in {'airport', 'airports'}):
            if 'AIRPORT' in upper:
                s += 0.5
        # Distance/miles
        if any(w in q_tokens for w in {'distance', 'miles', 'far'}):
            if 'DISTANCE' in upper or 'MILES' in upper:
                s += 1.0
        # Meal
        if any(w in q_tokens for w in {'meal', 'meals', 'food', 'dinner', 'lunch', 'breakfast', 'snack'}):
            if 'FOOD_SERVICE' in upper or 'MEAL' in upper:
                s += 1.0

        # EXECUTION-BASED VALIDATION
        recs, err = cached_execute(sql)
        if err:
            s -= 10.0
        elif len(recs) > 0:
            s += 3.0
            if len(recs) > 500:
                s -= 1.5  # overly broad
            if len(recs) > 1000:
                s -= 1.5
        else:
            s -= 0.5

        return s

    best_sql, _, _ = max(candidates, key=score_sql)
    if is_sql_like(best_sql):
        return best_sql
    return retrieval_idx.topk(question_text, k=1)[0]


def extract_question_text(text):
    lower = text.lower()
    q_key = 'question:'
    s_key = 'sql:'
    if q_key in lower:
        q_start = lower.index(q_key) + len(q_key)
        q_end = len(text)
        if s_key in lower[q_start:]:
            q_end = q_start + lower[q_start:].index(s_key)
        return text[q_start:q_end].strip()
    return text.strip()


def run_inference(checkpoint_dir, split, output_sql, output_pkl, batch_size=16):
    print(f"Loading model from {checkpoint_dir}")
    model = T5ForConditionalGeneration.from_pretrained(checkpoint_dir)
    model.to(DEVICE)
    model.eval()

    tok = T5TokenizerFast.from_pretrained('google-t5/t5-small')

    # Build retrieval index
    train_nl = load_lines('data/train.nl')
    train_sql = load_lines('data/train.sql')
    if split == 'test':
        dev_nl = load_lines('data/dev.nl')
        dev_sql = load_lines('data/dev.sql')
        retrieval_nl = train_nl + dev_nl
        retrieval_sql = train_sql + dev_sql
    else:
        retrieval_nl = train_nl
        retrieval_sql = train_sql
    retrieval_idx = RetrievalIndex(retrieval_nl, retrieval_sql)

    # Load data
    nl_path = f'data/{split}.nl'
    nl_lines = load_lines(nl_path)

    generated_sql_queries = []

    from load_data import get_dataloader
    if split == 'test':
        loader = get_dataloader(batch_size, 'test')
    else:
        loader = get_dataloader(batch_size, split)

    num_return_sequences = 10

    with torch.no_grad():
        if split == 'test':
            for encoder_input, encoder_mask, _ in tqdm(loader, desc=f"Inference ({split})"):
                encoder_input = encoder_input.to(DEVICE)
                encoder_mask = encoder_mask.to(DEVICE)

                generated = model.generate(
                    input_ids=encoder_input,
                    attention_mask=encoder_mask,
                    max_new_tokens=256,
                    num_beams=10,
                    num_return_sequences=num_return_sequences,
                    do_sample=False,
                    repetition_penalty=1.1,
                    length_penalty=1.0,
                    no_repeat_ngram_size=0,
                    early_stopping=True,
                )
                decoded = tok.batch_decode(generated, skip_special_tokens=True)
                source_texts = tok.batch_decode(encoder_input, skip_special_tokens=True)
                for i, src in enumerate(source_texts):
                    start = i * num_return_sequences
                    end = (i + 1) * num_return_sequences
                    cand_group = decoded[start:end]
                    qtext = extract_question_text(src)
                    generated_sql_queries.append(choose_best_sql(qtext, cand_group, retrieval_idx))
        else:
            for encoder_input, encoder_mask, _, decoder_targets, _ in tqdm(loader, desc=f"Inference ({split})"):
                encoder_input = encoder_input.to(DEVICE)
                encoder_mask = encoder_mask.to(DEVICE)

                generated = model.generate(
                    input_ids=encoder_input,
                    attention_mask=encoder_mask,
                    max_new_tokens=256,
                    num_beams=10,
                    num_return_sequences=num_return_sequences,
                    do_sample=False,
                    repetition_penalty=1.1,
                    length_penalty=1.0,
                    no_repeat_ngram_size=0,
                    early_stopping=True,
                )
                decoded = tok.batch_decode(generated, skip_special_tokens=True)
                source_texts = tok.batch_decode(encoder_input, skip_special_tokens=True)
                for i, src in enumerate(source_texts):
                    start = i * num_return_sequences
                    end = (i + 1) * num_return_sequences
                    cand_group = decoded[start:end]
                    qtext = extract_question_text(src)
                    generated_sql_queries.append(choose_best_sql(qtext, cand_group, retrieval_idx))

    # Enforce alignment
    expected_n = len(nl_lines)
    if len(generated_sql_queries) < expected_n:
        fill_sql = retrieval_idx.best_sql('')
        generated_sql_queries.extend([fill_sql] * (expected_n - len(generated_sql_queries)))
    elif len(generated_sql_queries) > expected_n:
        generated_sql_queries = generated_sql_queries[:expected_n]

    # Save
    from utils import save_queries_and_records
    os.makedirs(os.path.dirname(output_sql) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(output_pkl) or '.', exist_ok=True)
    save_queries_and_records(generated_sql_queries, output_sql, output_pkl)
    print(f"Saved {len(generated_sql_queries)} queries to {output_sql}")

    if split == 'dev':
        from utils import compute_metrics
        gt_sql = 'data/dev.sql'
        gt_pkl = 'records/dev_gt_records.pkl'
        sql_em, record_em, record_f1, errors = compute_metrics(gt_sql, output_sql, gt_pkl, output_pkl)
        error_rate = sum(1 for e in errors if e) / len(errors)
        print(f"Dev: SQL EM={sql_em:.4f}, Record EM={record_em:.4f}, Record F1={record_f1:.4f}, Error rate={error_rate*100:.1f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--split', type=str, default='dev', choices=['dev', 'test'])
    parser.add_argument('--output_sql', type=str, required=True)
    parser.add_argument('--output_pkl', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    run_inference(args.checkpoint, args.split, args.output_sql, args.output_pkl, args.batch_size)
