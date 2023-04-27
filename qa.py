import json
import random
import pandas as pd

NUM_SAMPLES = 10000
NUM_USING = 302
PATH = "nq_data.jsonl"
CONTEXT_WINDOW = 100


def get_question(sample):
    return sample['question_text']

def tokenize_answer(document_text):
    return document_text.split()

def format_data(tokens):
    return ' '.join(tokens)


def get_short_answer_and_context(sample):
    short_answers = sample['annotations'][0]['short_answers']
    yes_no_answer = sample['annotations'][0]['yes_no_answer']
    document_tokens = tokenize_answer(sample['document_text'])
    #only want samples that are not yes/no questions and have short answers
    if(len(short_answers) == 0 or yes_no_answer != "NONE"):
        return None
    
    SA = []
    contexts = []
    for short_answer in short_answers:
        start_index = short_answer['start_token']
        end_index = short_answer['end_token']
        context_start = max(0, start_index - CONTEXT_WINDOW)
        context_end = min(len(document_tokens)-1, end_index + CONTEXT_WINDOW)
        SA.append(format_data(document_tokens[start_index:end_index]))
        contexts.append(format_data(document_tokens[context_start:context_end]))
    return (SA, contexts)


def read_data(path):
    question_answer_pairs = []
    counter = 0
    with open("nq_data.jsonl", 'r') as f:
        for line in f:
            sample = json.loads(line)

            info = get_short_answer_and_context(sample)
            if info:
                
                question = get_question(sample)
                question_answer_pairs.append((question, info[0], info[1]))
                counter +=1
            
            if counter == NUM_SAMPLES:
                break
    return question_answer_pairs

questions_answers = read_data(PATH)
random.shuffle(questions_answers)
questions_answers = questions_answers[:NUM_USING]
questions = [triplet[0] for triplet in questions_answers]
answers = [triplet[1] for triplet in questions_answers]
contexts = [triplet[2] for triplet in questions_answers]
empty_arr = ["" for i in range(NUM_USING)]

qa_dict = {"Contexts": contexts, "Questions": questions, "Gold Answers": answers, "GPT Answer": empty_arr, "BERT Similarity GPT": empty_arr,
           "Base Model Answers": empty_arr, "BERT Similarity Baseline": empty_arr, "GPT Correctness":empty_arr, "Baseline Correctness": empty_arr}

qa_df = pd.DataFrame(qa_dict)

qa_df.to_csv("pairs.csv")






        
       
