import json
import random
import copy

import torch
import numpy as np


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

import torch
import torch.utils.data
import random

class OverpassDataset(torch.utils.data.Dataset):
    def __init__(self, nl, query, tokenizer, split, comments_dataset=None, inference_only=False, reverse=False, model_type='google/byt5-base'):
        self.reverse = reverse
        if self.reverse:
            self.input = query
            self.output = nl
        else:
            self.input = nl
            self.output = query
        self.tokenizer = tokenizer
        self.split = split
        self.comments_dataset = comments_dataset
        self.inference_only = inference_only
        # self.max_input_length = 0
        # self.max_output_length = 0
        # self.max_total_length = 0
        if model_type in ['google/byt5-small','google/byt5-base', 'google/byt5-large']:
            if self.split == "train" and not self.inference_only:
                # 99.9th percentile: 411
                # 4090 need 380
                # H20 411
                self.setting_max_lengths = 411
            elif self.split == "valid" and not self.inference_only:
                # max_lengths = max{input_lengths, output_lengths} = 578
                self.setting_max_lengths = 578
            elif self.split == "test" and not self.inference_only:
                # max_lengths = max{input_lengths, output_lengths} = 540
                self.setting_max_lengths = 540
            else:
                self.setting_max_lengths = 600
        else:
            self.setting_max_lengths = 512

    def __getitem__(self, idx):
        if idx >= len(self.input):
            if self.comments_dataset:
                idx = random.randint(0, len(self.comments_dataset) - 1)
                return self.comments_dataset[idx]

        input, output = self.input[idx], self.output[idx]

        model_inputs = self.tokenizer(
            input,
            max_length=self.setting_max_lengths,
            truncation=True,
        )
        
        # item = self.tokenizer(input)
        # item = self.tokenizer(input, max_length=512, truncation=True)
        if self.inference_only:
            return model_inputs
        
        model_inputs["labels"] = self.tokenizer(
            output,
            max_length=self.setting_max_lengths,
            truncation=True,
        ).input_ids

        # labels_with_ignore_index = [[label if label != 0 else -100 for label in label_set] for label_set in labels]
        
        # self.max_input_length = max(self.max_input_length, len(model_inputs["input_ids"]))
        # self.max_output_length = max(self.max_output_length, len(model_inputs["labels"]))
        # self.max_total_length = max(self.max_total_length, len(model_inputs["labels"]) + len(model_inputs["input_ids"]))

        return model_inputs

    def __len__(self):
        length = len(self.input)
        if self.comments_dataset:
            length += len(self.comments_dataset)  # Adjust the length if a comments dataset is included
        return length



def read_overpass_split(path, prefix_nl='',prefix_query='', num=None):
    nls = list()
    with open(path + '.nl') as f:
        for line in f:
            nls.append(prefix_nl + line.strip())

    queries = list()
    with open(path + '.query') as f:
        for line in f:
            queries.append(prefix_query + line.strip())

    assert len(nls) == len(queries)

    if num is not None:
        nls = nls[:num]
        queries = queries[:num]

    return nls, queries

def read_kv_split(path, prefix='', num=None,primary_kv=False):
    kvs = list()
    if primary_kv:
        print('use primary kv')
        with open(path + '.primary.kv') as f:
            for line in f:
                kvs.append(prefix + line.strip())
    else:
        print('use kv')
        with open(path + '.kv') as f:
            for line in f:
                kvs.append(prefix + line.strip())
    if num is not None:
        kvs = kvs[:num]
    return kvs
        

def get_comment_queries(path, comment_max_count=5, comment_min_content_ratio=0.85):
    comment_count = dict()
    instances = list()
    with open(path) as f:
        for line in f:
            instance = json.loads(line.strip())

            for comment in instance['comments']:
                if comment in comment_count:
                    comment_count[comment] += 1
                else:
                    comment_count[comment] = 1

            query = instance['query']
            if len(query) > 800:
                continue

            instances.append(instance)

    texts = list()
    labels = list()
    for instance in instances:
        comments = instance['comments']
        content_ratios = instance['comment_content_ratio']
        assert len(comments) == len(content_ratios)

        new_comments = list()
        for comment, content_ratio in zip(comments, content_ratios):
            if content_ratio < comment_min_content_ratio:
                continue
            if comment_count[comment] > comment_max_count:
                continue

            comment = comment.strip()
            if comment[-1] != '.':
                comment += '.'

            new_comments.append(comment)

        if len(new_comments) >= 1:

            text = ' '.join(new_comments)
            if len(text) < 800:
                texts.append(text)
                labels.append(instance['query'])

    return texts, labels

