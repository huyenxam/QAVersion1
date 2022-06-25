from msilib import sequence
import torch
import json
import numpy as np
from torch.utils.data import Dataset


class InputSample(object):
    def __init__(self, path, max_char_len=None, max_seq_length=None, stride=None):
        self.stride = stride
        self.max_char_len = max_char_len
        self.max_seq_length = max_seq_length
        self.list_sample = []
        with open(path, 'r', encoding='utf8') as f:
            self.list_sample = json.load(f)

    def get_character(self, word, max_char_len):
        word_seq = []
        for j in range(max_char_len):
            try:
                char = word[j]
            except:
                char = 'PAD'
            word_seq.append(char)
        return word_seq

    def get_sample(self):
        l_sample = []
        for i, sample in enumerate(self.list_sample):
            context = sample['context']
            question = sample['question'].split(' ')
            sample['question'] = question

            text_context = ""
            for item in context:
                text_context += " ".join(item) + " "
            text_context = text_context[:-1]

            sent = question + text_context
            sample['context'] = text_context

            char_seq = []
            for word in sent:
                character = self.get_character(word, self.max_char_len)
                char_seq.append(character)
            sample['char_sequence'] = char_seq

            len_ctx = 0
            sequence_idxs = []
            for ctx in context:
                if (len(ctx) + len_ctx) < (self.max_seq_length - len(question) - 2):
                    sequence_idxs.append([len_ctx, (len(ctx) + len_ctx -1)])
                len_ctx = len_ctx + len(ctx)
            
            sample['sequence_idxs'] = sequence_idxs
            label = sample['label']
            label_idxs = []
            if sent < self.max_seq_length:
                for lb in label:
                    ans_start = int(lb[1]) + len(question) + 2
                    ans_end = int(lb[2]) + len(question) + 2
                    entity = lb[0]
                    label_idxs.append([entity, ans_start, ans_end])
                sample['label_idx'] = label_idxs
                l_sample.append(sample)

        return l_sample


class MyDataSet(Dataset):

    def __init__(self, path, char_vocab_path, label_set_path,
                 max_char_len, tokenizer, max_seq_length, stride):

        self.samples = InputSample(path=path, max_char_len=max_char_len, max_seq_length=max_seq_length, stride=stride).get_sample()
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_char_len = max_char_len
        with open(label_set_path, 'r', encoding='utf8') as f:
            self.label_set = f.read().splitlines()

        with open(char_vocab_path, 'r', encoding='utf-8') as f:
            self.char_vocab = json.load(f)
        self.label_2int = {w: i for i, w in enumerate(self.label_set)}

    def preprocess(self, tokenizer, context, question, sequece_idxs, max_seq_length, mask_padding_with_zero=True):
        firstSWindices = []
        input_ids = [tokenizer.cls_token_id]
        for i in range(len(firstSWindices)):
            firstSWindices[i].append(len(input_ids))

        for w in question:
            word_token = tokenizer.encode(w)
            input_ids += word_token[1: (len(word_token) - 1)]
            for i in range(len(firstSWindices)):
                firstSWindices[i].append(len(input_ids))
        
        input_ids.append(tokenizer.sep_token_id)
        for i in range(len(firstSWindices)):
            firstSWindices[i].append(len(input_ids))

        for i, w in enumerate(context):
            word_token = tokenizer.encode(w)
            input_ids += word_token[1: (len(word_token) - 1)]
            for j in range(len(firstSWindices)):
                if i >= sequece_idxs[j][0] and i <= sequece_idxs[j][1]:
                    if len(input_ids) >= max_seq_length:
                        firstSWindices[j].append(0)
                    else:
                        firstSWindices[j].append(len(input_ids))

        input_ids.append(tokenizer.sep_token_id)
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        for i in range(len(firstSWindices)):
            if len(firstSWindices[i]) > max_seq_length:
                firstSWindices[i] = firstSWindices[i][:max_seq_length]
            else:
                firstSWindices[i] = firstSWindices[i] + [0] * (max_seq_length - len(firstSWindices[i]))

        if len(input_ids) > max_seq_length:
            input_ids = input_ids[:max_seq_length]
            attention_mask = attention_mask[:max_seq_length]
        else:
            attention_mask = attention_mask + [0 if mask_padding_with_zero else 1] * (max_seq_length - len(input_ids))
            input_ids = (
                    input_ids
                    + [
                        tokenizer.pad_token_id,
                    ]
                    * (max_seq_length - len(input_ids))
            )

        return torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(firstSWindices)

    def character2id(self, character_sentence, max_seq_length):
        char_ids = []
        for word in character_sentence:
            word_char_ids = []
            for char in word:
                if char not in self.char_vocab:
                    word_char_ids.append(self.char_vocab['UNK'])
                else:
                    word_char_ids.append(self.char_vocab[char])
            char_ids.append(word_char_ids)
        if len(char_ids) < max_seq_length:
            char_ids += [[self.char_vocab['PAD']] * self.max_char_len] * (max_seq_length - len(char_ids))
        return torch.tensor(char_ids)

    def span_maxtrix_label(self, label):
        start, end, entity = [], [], []
        label = np.unique(label, axis=0).tolist()
        for lb in label:
            if int(lb[1]) > self.max_seq_length or int(lb[2]) > self.max_seq_length:
                start.append(0)
                end.append(0)
            else:
                start.append(int(lb[1]))
                end.append(int(lb[2]))
            try:
                entity.append(self.label_2int[lb[0]])
            except:
                print(lb)
        
        label = torch.sparse.FloatTensor(torch.tensor([start, end], dtype=torch.int64), torch.tensor(entity),
                                         torch.Size([self.max_seq_length, self.max_seq_length])).to_dense()
        
        return label

    def __getitem__(self, index):

        sample = self.samples[index]
        context = sample['context']
        question = sample['question']
        char_seq = sample['char_sequence']
        sequece_idxs = sample['sequece_idxs']
        seq_length = len(question) + len(context) + 2        
        label = sample['label_idx']
        input_ids, attention_mask, firstSWindices = self.preprocess(self.tokenizer, context, question, sequece_idxs, self.max_seq_length)

        char_ids = self.character2id(char_seq, max_seq_length=self.max_seq_length)
        if seq_length > self.max_seq_length:
          seq_length = self.max_seq_length
        label = self.span_maxtrix_label(label)

        return input_ids, attention_mask, firstSWindices, torch.tensor([seq_length]), char_ids, label.long()

    def __len__(self):
        return len(self.samples)


def get_mask(max_length, seq_length):
    mask = [[1] * seq_length[i] + [0] * (max_length - seq_length[i]) for i in range(len(seq_length))]
    mask = torch.tensor(mask)
    mask = mask.unsqueeze(1).expand(-1, mask.shape[-1], -1)
    mask = torch.triu(mask)
    return mask


def get_useful_ones(out, label, mask):
    # get mask, mask the padding and down triangle

    mask = mask.reshape(-1)
    tmp_out = out.reshape(-1, out.shape[-1])
    tmp_label = label.reshape(-1)
    # index select, for gpu speed
    indices = mask.nonzero(as_tuple=False).squeeze(-1).long()
    tmp_out = tmp_out.index_select(0, indices)
    tmp_label = tmp_label.index_select(0, indices)

    return tmp_out, tmp_label