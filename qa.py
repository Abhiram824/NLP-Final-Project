import json
from bs4 import BeautifulSoup
import csv
import random

import pandas as pd

with open('data.jsonl', 'r') as f:
    pairs = []
    count = 0
    for line in f:
        if count >= 2000: #TODO CHANGE TO 10000
            break
        if count%10 == 0:
            json_obj = json.loads(line)

            question_text = json_obj['question_text']

            answer_start_token = json_obj['annotations'][0]['long_answer']['start_token']
            answer_end_token = json_obj['annotations'][0]['long_answer']['end_token']
            document_tokens = json_obj['document_tokens']
            answer_tokens = document_tokens[answer_start_token:answer_end_token+1]
            answer_text = ''
            allowed = [',', '\'', '!', '.', '?', ':', ';', ')', ']', '}']
            for i in range(len(answer_tokens)-1):
                token = answer_tokens[i]['token']
                next_token = answer_tokens[i+1]['token']

                if next_token in allowed:
                    answer_text += token
                else:
                    if token in '([{':
                        answer_text += token
                    else:
                        answer_text += token + ' '

            if len(answer_text) == 0:
                continue
            else:
                answer_text = answer_text.rstrip()
                answer_text += answer_tokens[-1]['token']

            soup = BeautifulSoup(answer_text, 'html.parser')
            isBad = False
            allowed_tags = {'p'}
            for tag in soup.find_all():
                if tag.name in allowed_tags:
                    tag.extract()
                if tag.name not in allowed_tags:
                    isBad = True

            if isBad:
                continue
            else:
                answer_text = answer_text.replace('<P>', '')
                answer_text = answer_text.replace('</P>', '')

            # Print the question-answer pair
            pairs.append((question_text, answer_text))
        count += 1

random.shuffle(pairs)
pairs = pairs[:125]

questions = [pair[0] for pair in pairs]
answers = [pair[1] for pair in pairs]
empty_arr = ["" for _ in range(len(pairs))]

qa_dict = {"Questions": questions, "Gold Answers": answers, "GPT Answer": empty_arr, "BERT Similarity GPT": empty_arr,
           "Base Model Answers": empty_arr, "BERT Similarity Baseline": empty_arr, "GPT Correctness":empty_arr, "Baseline Correctness": empty_arr}

qa_df = pd.DataFrame(qa_dict)

qa_df.to_csv("pairs.csv")

#TODO add columns names
# with open('pairs.csv', 'a', encoding='utf-8', newline='') as csvfile:
#     # Create a writer object
#     writer = csv.writer(csvfile)
    
#     # Write each tuple to the CSV file
#     for row in pairs:
#         writer.writerow(row)