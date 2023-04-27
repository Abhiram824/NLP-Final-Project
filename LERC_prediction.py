from allennlp.predictors import Predictor
from lerc.lerc_predictor import LERCPredictor
import pandas as pd
from ast import literal_eval

predictor = Predictor.from_path(
    archive_path='https://storage.googleapis.com/allennlp-public-models/lerc-2020-11-18.tar.gz',
    predictor_name='lerc',
    cuda_device=0
)

def LERC_score(context, question, reference, candidate):
    
    input_json = {
        'context': context,
        'question': question,
        'reference': reference, 
        'candidate': candidate
    }

    output_dict = predictor.predict_json(input_json)
    return output_dict['pred_score']

def compute_LERC_score(contexts, question, references, candidate):
    assert len(contexts) == len(references)
    return max([LERC_score(context, question, reference, candidate) for context, reference in zip(contexts, references)])

combined_pd = pd.read_csv("answer_pairs.csv")
combined_pd["Gold Answers"] = combined_pd["Gold Answers"].apply(literal_eval)
combined_pd["Contexts"] = combined_pd["Contexts"].apply(literal_eval)
combined_pd["LERC Score"] = combined_pd.apply(lambda row: compute_LERC_score(row["Contexts"], row['Questions'], row['Gold Answers'], row['GPT Answer']), axis=1)
combined_pd.to_csv("combined.csv")



