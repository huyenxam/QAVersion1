import json
from metrics.f1_score import f1_score
from metrics.exact_match_score import exact_match_score
from dataloader import *
import numpy as np

def evaluate(outputs, max_char_len, max_seq_length, path):


    f1 = exact_match = 0 
    list_sample = []
    with open(path, 'r', encoding='utf8') as f:
        list_sample = json.load(f)

    i = 0
    # Lặp qua từng mẫu
    for sample in list_sample:
        # Lấy context của từng sample
        context = sample['context']
        question = sample['question'].split(" ")
        text_context = ""
        for item in context:
            text_context += " ".join(item) + " "
        text_context = text_context[:-1].split(' ')

        label_prediction = ""
        score_max = 0
        # Lặp qua từng câu trong context
        len_ctx = 0
        for ctx in context:
            if (len_ctx + len(ctx)) < max_seq_length:
                sentence = ['cls'] + question + ['sep'] +  ctx
                if score_max < outputs[i][3]:      
                    score_max = outputs[i][3]
                    start_pre = outputs[i][1]
                    end_pre = outputs[i][2]
                    label_prediction = " ".join(sentence[start_pre:end_pre+1])
                i += 1
            len_ctx = len_ctx + len(ctx)
        # Lấy câu trả lời trong từng sample
        labels = sample['label']
        f1_idx = [0]
        extract_match_idx = [0]
        for lb in labels:
            start = int(lb[1])
            end = int(lb[2])
            ground_truth = " ".join(text_context[start:end+1])
            f1_idx.append(f1_score(label_prediction, ground_truth))
            extract_match_idx.append(exact_match_score(label_prediction, ground_truth))
            # print(ground_truth)
            # print(label_prediction)

        f1 += max(f1_idx)
        exact_match += max(extract_match_idx)    



    # output = np.zeros(20000) 
    # inputs = InputSample(path=path, max_char_len=max_char_len, max_seq_length=max_seq_length).get_sample()

    # j = -1
    # label_prediction = ""
    # idx = 0
    # for i, sample in enumerate(inputs):
    #     idx = sample['sample']
    #     context = sample['context']
    #     question = sample['question']
    #     sentence = ['cls'] + question + ['sep'] +  context

    #     if idx > j:
    #         j = idx
    #         output[idx] = prediction[i][3]
    #         start_pre = prediction[i][1]
    #         end_pre = prediction[i][2]
    #         if idx > 0:
    #             f1_idx = [0]
    #             extract_match_idx = [0]
    #             answers = inputs[i-1]['answer']
    #             for ans in answers:
    #                 # print(label_prediction)
    #                 # print(ans)
    #                 f1_idx.append(f1_score(label_prediction, ans))
    #                 extract_match_idx.append(exact_match_score(label_prediction, ans))

    #             f1 += max(f1_idx)
    #             exact_match += max(extract_match_idx)

    #         label_prediction = " ".join(sentence[start_pre:end_pre+1])
    #     else:
    #         if output[idx] < prediction[i][3]:
    #             output[idx] = prediction[i][3]
    #             start_pre = prediction[i][1]
    #             end_pre = prediction[i][2]
    #             label_prediction = " ".join(sentence[start_pre:end_pre+1])

    total = len(list_sample)
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    
    return exact_match, f1