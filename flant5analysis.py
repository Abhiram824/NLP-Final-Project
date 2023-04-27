import pandas as pd

df = pd.read_csv('flan_answers_100.csv')

label_freq = {}
category_freq = {}

for index, row in df.iterrows():
    # only annotated first 100
    if index == 100:
        break

    label = int(row['Annotator Label'])
    category = row['Category']

    if label not in label_freq:
        label_freq[label] = 0
    
    if category not in category_freq:
        category_freq[category] = 0

    label_freq[label] += 1
    category_freq[category] += 1

print('Label distribution:', label_freq)
print('Category distribution:', category_freq)