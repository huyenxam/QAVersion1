from operator import le
import torch
import json
import numpy as np
from torch.utils.data import Dataset


class InputSampleTrain(object):
    def __init__(self, path, max_char_len=None, max_seq_length=None, max_ctx_length=None):
        self.max_char_len = max_char_len
        self.max_seq_length = max_seq_length
        self.max_ctx_length = max_ctx_length
        self.list_sample = []
        with open(path, 'r', encoding='utf8') as f:
            self.list_sample = json.load(f)
        # self.list_sample = self.list_sample[:10]
        
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
        for sample in self.list_sample:
            text_question = sample['question'].split(' ')
            
            context = sample['context']
            text_context = ""
            for item in context:
              text_context += " ".join(item) + " "
            text_context = text_context[:-1].split(' ')

            length_ctx = self.max_seq_length - len(text_question) - 2
            if len(text_context) > length_ctx:
                text_context = text_context[:length_ctx]

            sent = text_question + text_context
            char_seq = []
            for word in sent:
                character = self.get_character(word, self.max_char_len)
                char_seq.append(character)

            len_ctx = 0
            list_context = []
            idx = 0
            i = 0
            for ctx in context:
                if (len_ctx + len(ctx)) < self.max_seq_length:
                    qa_dict = {}
                    qa_dict['question'] = text_question
                    qa_dict['context'] = text_context
                    qa_dict['char_sequence'] = char_seq
                    qa_dict['sequence_idx'] = [len_ctx + len(text_question) + 2, len_ctx + len(ctx) + len(text_question) + 1]
                    qa_dict['seq_length'] = len(ctx)

                    label_list = []
                    label = sample['label'][0]
                    entity = label[0]
                    start = int(label[1])
                    end = int(label[2])                    
                    if start >= len_ctx and end <= (len_ctx + len(ctx) - 1):
                        start = start - len_ctx + len(text_question) + 2
                        end = end - len_ctx + len(text_question) + 2
                        idx = i
                        if end > self.max_seq_length:
                            end = self.max_seq_length - 1
                        label_list.append([entity, start, end])
                        
                    qa_dict['label_idx'] = label_list
                    list_context.append(qa_dict)
                    i = i + 1
                len_ctx = len_ctx + len(ctx)

            try:
                if idx == (len(context) - 1):
                    # l_sample.append(list_context[idx - 3]) 
                    l_sample.append(list_context[idx - 2]) 
                    l_sample.append(list_context[idx - 1]) 
                    l_sample.append(list_context[idx]) 
                elif idx == 0:
                    # l_sample.append(list_context[idx + 3]) 
                    l_sample.append(list_context[idx + 2]) 
                    l_sample.append(list_context[idx + 1]) 
                    l_sample.append(list_context[idx])
                else:
                    # l_sample.append(list_context[idx - 1]) 
                    l_sample.append(list_context[idx]) 
                    l_sample.append(list_context[idx + 1]) 
                    l_sample.append(list_context[idx + 2])
            except:
                    l_sample.append(list_context[idx]) 

        return l_sample


class MyDataSetTrain(Dataset):

    def __init__(self, path, char_vocab_path, label_set_path,
                 max_char_len, tokenizer, max_seq_length, max_ctx_length):

        self.samples = InputSampleTrain(path=path, max_char_len=max_char_len, max_seq_length=max_seq_length, max_ctx_length=max_ctx_length).get_sample()
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_char_len = max_char_len
        self.max_ctx_length = max_ctx_length
        with open(label_set_path, 'r', encoding='utf8') as f:
            self.label_set = f.read().splitlines()

        with open(char_vocab_path, 'r', encoding='utf-8') as f:
            self.char_vocab = json.load(f)
        self.label_2int = {w: i for i, w in enumerate(self.label_set)}

    def preprocess(self, tokenizer, context, question, sequence_idx, max_seq_length, max_ctx_length, mask_padding_with_zero=True):
        firstSWindices = [0]
        input_ids = [tokenizer.cls_token_id]
        firstSWindices.append(len(input_ids))

        for w in question:
            word_token = tokenizer.encode(w)
            input_ids += word_token[1: (len(word_token) - 1)]
            firstSWindices.append(len(input_ids))
        
        input_ids.append(tokenizer.sep_token_id)
        firstSWindices.append(len(input_ids))

        start_ctx = sequence_idx[0]
        end_ctx = sequence_idx[1]
        for i, w in enumerate(context):
            word_token = tokenizer.encode(w)
            input_ids += word_token[1: (len(word_token) - 1)]
            if start_ctx <= i and i <= end_ctx:
                if len(input_ids) >= max_seq_length:
                    firstSWindices.append(0)
                else:
                    firstSWindices.append(len(input_ids))

        input_ids.append(tokenizer.sep_token_id)
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        if len(firstSWindices) > max_ctx_length:
            firstSWindices = firstSWindices[:max_ctx_length]
        else:
            firstSWindices = firstSWindices + [0] * (max_ctx_length - len(firstSWindices))

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

    def character2id(self, character_sentence, max_ctx_length):
        char_ids = []
        for word in character_sentence:
            word_char_ids = []
            for char in word:
                if char not in self.char_vocab:
                    word_char_ids.append(self.char_vocab['UNK'])
                else:
                    word_char_ids.append(self.char_vocab[char])
            char_ids.append(word_char_ids)
        if len(char_ids) < max_ctx_length:
            char_ids += [[self.char_vocab['PAD']] * self.max_char_len] * (max_ctx_length - len(char_ids))
        else:
            char_ids = char_ids[:max_ctx_length]
        return torch.tensor(char_ids)

    def span_maxtrix_label(self, label):
        start, end, entity = [], [], []
        label = np.unique(label, axis=0).tolist()
        for lb in label:
            if int(lb[1]) > self.max_ctx_length or int(lb[2]) > self.max_ctx_length:
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
                                         torch.Size([self.max_ctx_length, self.max_ctx_length])).to_dense()
        
        return label

    def __getitem__(self, index):

        sample = self.samples[index]
        context = sample['context']
        question = sample['question']
        char_seq = sample['char_sequence']
        seq_length = sample['seq_length']       
        label = sample['label_idx']
        sequence_idx = sample['sequence_idx']
        input_ids, attention_mask, firstSWindices = self.preprocess(self.tokenizer, context, question, sequence_idx, self.max_seq_length, self.max_ctx_length)

        char_ids = self.character2id(char_seq, max_ctx_length=self.max_ctx_length)
        if seq_length > self.max_ctx_length:
          seq_length = self.max_ctx_length
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