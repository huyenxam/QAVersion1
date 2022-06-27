import json
from metrics.f1_score import f1_score
from metrics.exact_match_score import exact_match_score
from dataloader import *
import numpy as np

def evaluate(prediction, max_char_len, max_seq_length, path):
    # if mode == 'dev':
    #     path = 'data/Data/dev_ViQuAD.json'
    # elif mode == 'test':
    #     path = 'data/Data/test_ViQuAD.json'
    # else:
    #     raise Exception("Only dev and test dataset available")

    f1 = exact_match = 0        
    output = np.zeros(20000) 
    inputs = InputSample(path=path, max_char_len=max_char_len, max_seq_length=max_seq_length).get_sample()

    j = -1
    label_prediction = ""
    idx = 0
    for i, sample in enumerate(inputs):
        idx = sample['sample']
        context = sample['context']
        sequence = sample['sequence']
        question = sample['question']
        sentence = ['cls'] + question + ['sep'] +  sequence[idx]

        if idx > j:
            j = idx
            output[idx] = prediction[i][3]
            start_pre = prediction[i][1]
            end_pre = prediction[i][2]
            if idx > 0:
                f1_idx = [0]
                extract_match_idx = [0]
                answers = inputs[i-1]['answer']
                for ans in answers:
                    # print(label_prediction)
                    # print(ans)
                    f1_idx.append(f1_score(label_prediction, ans))
                    extract_match_idx.append(exact_match_score(label_prediction, ans))

                f1 += max(f1_idx)
                exact_match += max(extract_match_idx)

            label_prediction = " ".join(sentence[start_pre:end_pre+1])
        else:
            if output[idx] < prediction[i][3]:
                output[idx] = prediction[i][3]
                start_pre = prediction[i][1]
                end_pre = prediction[i][2]
                label_prediction = " ".join(sentence[start_pre:end_pre+1])

    total = idx + 1
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    
    return exact_match, f1