import pandas as pd
import numpy as np

am_df = pd.read_csv('answer_pairs-am.csv')
ae_df = pd.read_csv('answer_pairs-ae.csv')
nv_df = pd.read_csv('answer_pairs-nv.csv')

am_gpt = am_df['GPT Correctness'][:25]
ae_gpt = ae_df['GPT Correctness'][:25]
nv_gpt = nv_df['GPT Correctness'][:25]

print(am_gpt.head())
print(ae_gpt.head())
print(nv_gpt.head())

def check_same_annotation(all_gpt_annotations, annotation_idx):
    for j in range(len(all_gpt_annotations)-1):
        if all_gpt_annotations[j][annotation_idx] != all_gpt_annotations[j+1][annotation_idx]:
            return False
    return True

def get_agreement(all_gpt_annotations):
    n = len(all_gpt_annotations[0])
    num_same = 0
    # iterate through each annotation
    for i in range(n):
        # make sure each person has same annotation
        if check_same_annotation(all_gpt_annotations, i):
            num_same += 1
    return num_same/n

def get_distribution(all_gpt_annotations):
    for gpt_annotations in all_gpt_annotations:
        freq = {}
        for a in gpt_annotations:
            if a not in freq:
                freq[a] = 0
            freq[a] += 1
        print(freq)

# am to ae
print('agreement am-ae:', get_agreement([am_gpt, ae_gpt]))

# ae to nv
print('agreement ae-nv:', get_agreement([ae_gpt, nv_gpt]))

# am to nv
print('agreement am-nv:', get_agreement([am_gpt, nv_gpt]))

# am to ae to nv
print('agreement am-ae-nv:', get_agreement([am_gpt, ae_gpt, nv_gpt]))

# label distributions
get_distribution([am_gpt, ae_gpt, nv_gpt])