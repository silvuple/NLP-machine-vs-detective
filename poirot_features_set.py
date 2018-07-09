import pronovel
import pandas as pd

titles = []
with open('poirot_titles.csv') as f:
    for line in f:
        titles.append('novels/' + line.split(';')[0] + '.txt')

characters = []
df = pd.DataFrame()
for title in titles[:1]:
    try:
        file = open(title, encoding='utf8')
        raw = file.read()
        file.close()
    except OSError:
        print('file "{}" not found'.format(title.split('/')[1]))
        continue
    text = pronovel.NovelText(raw)
    characters.append(text.get_persons())

##df = pd.DataFrame(characters, columns=['Character_name'])
##df.to_csv('features_set.csv')
    
    
        
