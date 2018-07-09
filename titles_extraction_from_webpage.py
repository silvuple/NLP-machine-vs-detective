from requests import get
import re
import pandas as pd
from bs4 import BeautifulSoup


titles_list = []
url = 'https://en.wikipedia.org/wiki/Hercule_Poirot_in_literature'

# get url webpage using requests HTTP library for Python
r = get(url)

# parse the webpage using BeautifulSoup's html.parser
soup = BeautifulSoup(r.text, 'html.parser')

# find all listed book titles 
results = soup.ol.find_all('li')

# iterate over the results list and extract data
for result in results:
    # get title
    title = result.i.text

    # get publication year
    year = re.search(r'\d{4}', result.text)
    year = year.group()

    # mark if 'short stories collection' s(ss)
    short_stories = ''
    if re.search(r'\(\d{4},\s(<i>)*ss(<i>)*\)', result.text):
        short_stories = 'Y'
        
    # add (title, year, alt_title, short_stories) to the list
    titles_list.append((title, year, short_stories))

# create pandas DataFrame from the titles list    
df = pd.DataFrame(titles_list, columns=['Title', 'Year', 'Short_Stories'])

# export DataFrame to csv
df.to_csv('poirot_titles.csv', sep=';', header=None, index=False)

# get extended titles list including titles of short stories
extended_list = []

# the lists of titles starts after the <h2><span> tag with below id string
start_tag = soup.find('span', id="Books_in_chronological_order")
results2 = start_tag.find_all_next('li')

# for each list item (tag <li>) check if it has <a> tag with attribute
# href that starts with string '/wiki/', if it does - this is the title
for result in results2:
    try:
        ex_title = result.find('a', href=re.compile(r'^/wiki/')).text
        extended_list.append(ex_title)
    except:
        continue
# last title is "Curtain", data beyond this title is irrelevant
last_index = extended_list.index('Curtain')+1
extended_list = extended_list[:last_index]
    
# create pandas DataFrame from the extended titles list    
df = pd.DataFrame(extended_list, columns=['Title'])

# export DataFrame to csv
df.to_csv('poirot_extended_titles_list.txt', index=False, header=False)
