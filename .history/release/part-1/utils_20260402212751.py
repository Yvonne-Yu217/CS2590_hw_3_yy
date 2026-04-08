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

    # Lowercase the text first
    text = example["text"].strip()

    # In case wordnet is not downloaded yet, try to download it
    try:
        from nltk.corpus import wordnet
    except Exception:
        import nltk
        nltk.download('wordnet', quiet=True)
        from nltk.corpus import wordnet

    tokens = word_tokenize(text)
    transformed_tokens = []

    for token in tokens:
        if random.random() < 0.15 and token.isalpha():
            synsets = wordnet.synsets(token)
            synonyms = set()
            for syn in synsets:
                for lemma in syn.lemmas():
                    w = lemma.name().replace('_', ' ')
                    if w.lower() != token.lower():
                        synonyms.add(w)
            if synonyms:
                new_word = random.choice(list(synonyms))
                transformed_tokens.append(new_word)
                continue

            # Fallback to a small keyboard typo substitution
            if len(token) > 1:
                i = random.randrange(len(token))
                if i == 0:
                    j = 1
                else:
                    j = i - 1
                token_list = list(token)
                token_list[i], token_list[j] = token_list[j], token_list[i]
                transformed_tokens.append(''.join(token_list))
                continue

        transformed_tokens.append(token)

    transformed_text = TreebankWordDetokenizer().detokenize(transformed_tokens)
    example["text"] = transformed_text

    ##### YOUR CODE ENDS HERE ######

    return example
