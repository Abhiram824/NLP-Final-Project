from sentence_transformers import SentenceTransformer, util
import pandas as pd

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def calc_similarity(answer1, answer2):
    embeddings1 = model.encode(answer1, convert_to_tensor=True)
    embeddings2 = model.encode(answer2, convert_to_tensor=True)
    cosine_score = util.cos_sim(embeddings1, embeddings2)
    return cosine_score[0][0].item()

df = pd.read_csv("answer_pairs.csv")
df['BERT Similarity GPT'] = df.apply(lambda row: calc_similarity(row['Gold Answers'], row['GPT Answer']), axis=1)
df.to_csv("similarities.csv", index=False)



