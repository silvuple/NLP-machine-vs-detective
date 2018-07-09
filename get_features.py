import pronovel
import pandas as pd

titles = []
with open('poirot_titles.csv') as f:
    for line in f:
        titles.append('novels/' + line.split(';')[0] + '.txt')

recurring_characters = ['Hercule', 'Poirot', 'Hastings',
                        'Ariadne', 'Mrs. Oliver', 'Miss Lemon',
                        'Inspector Japp', 'Japp', 'Inspector']
all_texts_features = {}
for title in titles:
    try:
        file = open(title, encoding='utf8')
        raw = file.read()
        file.close()
    except OSError:
        print('file "{}" not found'.format(title.split('/')[1]))
        continue
    text = pronovel.NovelText(raw)
    text_feat = {}
    text_persons = text.get_persons()
    text_persons_count = text.get_persons_count()
    text_persons_gender = text.get_persons_gender()
    for person in text_persons:
        text_feat.setdefault(person, []).append(text_persons_count[person])
        text_feat[person].append(text_persons_gender[person])
    all_texts_features = {**all_texts_features, **text_feat}

df = pd.DataFrame.from_dict(all_texts_features, orient='index',
                            columns=['Character_count', 'Gender'])
print(df.head())
df.to_csv('features_set.csv')
    
    
        
