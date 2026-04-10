import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    text = example["text"].strip()
    tokens = word_tokenize(text)
    transformed_tokens = []

    # A stronger but still realistic perturbation: lexical substitutions + keyboard typos.
    token_edit_prob = 0.35
    keyboard_neighbors = {
        'a': 'qwsz', 'b': 'vghn', 'c': 'xdfv', 'd': 'ersfcx', 'e': 'rdsw',
        'f': 'rtgdvc', 'g': 'tyfhvb', 'h': 'yugjbn', 'i': 'uojk', 'j': 'uikhmn',
        'k': 'iojlm', 'l': 'opk', 'm': 'njk', 'n': 'bhjm', 'o': 'pikl',
        'p': 'ol', 'q': 'wa', 'r': 'tfde', 's': 'wedxza', 't': 'ygfr',
        'u': 'ijhy', 'v': 'cfgb', 'w': 'qeas', 'x': 'zsdc', 'y': 'uhtg',
        'z': 'asx'
    }

    def preserve_case(src, tgt):
        if src.isupper():
            return tgt.upper()
        if len(src) > 0 and src[0].isupper():
            return tgt.capitalize()
        return tgt

    def typo_token(tok):
        if len(tok) <= 1:
            return tok
        chars = list(tok)
        op = random.choice(['swap', 'neighbor', 'delete'])
        if op == 'swap' and len(chars) > 1:
            i = random.randint(0, len(chars) - 2)
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
            return ''.join(chars)
        if op == 'neighbor':
            alpha_positions = [i for i, c in enumerate(chars) if c.lower() in keyboard_neighbors]
            if alpha_positions:
                i = random.choice(alpha_positions)
                c = chars[i].lower()
                repl = random.choice(keyboard_neighbors[c])
                chars[i] = repl.upper() if chars[i].isupper() else repl
                return ''.join(chars)
        if op == 'delete' and len(chars) > 3:
            i = random.randint(1, len(chars) - 2)
            del chars[i]
            return ''.join(chars)
        return tok

    for token in tokens:
        if token.isalpha() and random.random() < token_edit_prob:
            # Try synonym substitution first (reasonable semantic perturbation).
            synsets = wordnet.synsets(token)
            synonyms = []
            seen = set()
            for syn in synsets:
                for lemma in syn.lemmas():
                    candidate = lemma.name().replace('_', ' ')
                    if ' ' in candidate:
                        continue
                    if candidate.lower() == token.lower():
                        continue
                    if candidate.lower() in seen:
                        continue
                    seen.add(candidate.lower())
                    synonyms.append(candidate)

            if synonyms and random.random() < 0.7:
                replacement = random.choice(synonyms)
                transformed_tokens.append(preserve_case(token, replacement))
            else:
                transformed_tokens.append(typo_token(token))
        else:
            transformed_tokens.append(token)

    example["text"] = TreebankWordDetokenizer().detokenize(transformed_tokens)

    ##### YOUR CODE ENDS HERE ######

    return example
