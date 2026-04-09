import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.data_folder = data_folder
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        self.bos_token_id = self.tokenizer.pad_token_id
        self.examples = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        nl_path = os.path.join(data_folder, f'{split}.nl')
        nl_lines = load_lines(nl_path)

        sql_lines = None
        if split != 'test':
            sql_path = os.path.join(data_folder, f'{split}.sql')
            sql_lines = load_lines(sql_path)

        max_encoder_len = 256
        max_decoder_len = 256
        examples = []

        for i, nl in enumerate(nl_lines):
            # Make task intent explicit and end with an SQL cue for decoder behavior.
            prompted_nl = f"Translate the question to SQL.\nQuestion: {nl}\nSQL:"
            enc = tokenizer(
                prompted_nl,
                add_special_tokens=True,
                truncation=True,
                max_length=max_encoder_len,
            )

            encoder_ids = torch.tensor(enc['input_ids'], dtype=torch.long)
            encoder_mask = torch.tensor(enc['attention_mask'], dtype=torch.long)

            if split == 'test':
                examples.append({
                    'encoder_ids': encoder_ids,
                    'encoder_mask': encoder_mask,
                    'initial_decoder_input': torch.tensor([self.bos_token_id], dtype=torch.long),
                })
                continue

            sql = sql_lines[i]
            dec = tokenizer(
                sql,
                add_special_tokens=False,
                truncation=True,
                max_length=max_decoder_len - 1,
            )

            target_ids = dec['input_ids'] + [tokenizer.eos_token_id]
            decoder_targets = torch.tensor(target_ids, dtype=torch.long)
            decoder_inputs = torch.tensor([self.bos_token_id] + target_ids[:-1], dtype=torch.long)

            examples.append({
                'encoder_ids': encoder_ids,
                'encoder_mask': encoder_mask,
                'decoder_inputs': decoder_inputs,
                'decoder_targets': decoder_targets,
                'initial_decoder_input': torch.tensor([self.bos_token_id], dtype=torch.long),
            })

        return examples
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        if self.split == 'test':
            return ex['encoder_ids'], ex['encoder_mask'], ex['initial_decoder_input']
        return ex['encoder_ids'], ex['encoder_mask'], ex['decoder_inputs'], ex['decoder_targets'], ex['initial_decoder_input']

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids = [x[0] for x in batch]
    encoder_masks = [x[1] for x in batch]
    decoder_inputs = [x[2] for x in batch]
    decoder_targets = [x[3] for x in batch]
    initial_decoder_inputs = [x[4] for x in batch]

    encoder_ids = pad_sequence(encoder_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_masks = pad_sequence(encoder_masks, batch_first=True, padding_value=0)
    decoder_inputs = pad_sequence(decoder_inputs, batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence(decoder_targets, batch_first=True, padding_value=PAD_IDX)
    initial_decoder_inputs = torch.stack(initial_decoder_inputs, dim=0)

    return encoder_ids, encoder_masks, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids = [x[0] for x in batch]
    encoder_masks = [x[1] for x in batch]
    initial_decoder_inputs = [x[2] for x in batch]

    encoder_ids = pad_sequence(encoder_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_masks = pad_sequence(encoder_masks, batch_first=True, padding_value=0)
    initial_decoder_inputs = torch.stack(initial_decoder_inputs, dim=0)

    return encoder_ids, encoder_masks, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    train_x = load_lines(os.path.join(data_folder, 'train.nl'))
    train_y = load_lines(os.path.join(data_folder, 'train.sql'))
    dev_x = load_lines(os.path.join(data_folder, 'dev.nl'))
    dev_y = load_lines(os.path.join(data_folder, 'dev.sql'))
    test_x = load_lines(os.path.join(data_folder, 'test.nl'))
    return train_x, train_y, dev_x, dev_y, test_x