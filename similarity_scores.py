import pandas as pd
import numpy as np

df = pd.read_csv('answer_pairs-am.csv')
similarity_scores = df['BERT Similarity GPT'][:25]
avg_score = np.mean(similarity_scores)

print('Average cosine similarity:', avg_score)