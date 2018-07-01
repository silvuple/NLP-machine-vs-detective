import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk_sents
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.draw import dispersion_plot


with open('the_mystirious_affair_at_styles.txt') as f:
    raw = f.read()

# Break text to tokens list (words, punctutaions etc.)
tokens = word_tokenize(raw)

# Create words list (exclude punctuations and convert to lowercase)
words = [w.lower() for w in tokens if w[0].isalnum()]

# Create vocabulary list (list of unique words).
vocab = sorted(set(words))

# Break text to sentences list.
sentences = sent_tokenize(raw)

# Create tokenized sentences list (list of lists of lists).
sentences = [word_tokenize(sent) for sent in sentences]

# Add Part-of-Speach tagging to tokenized sentences.
sentences = [pos_tag(sent) for sent in sentences]

# Create Frequency Distribution obejct.
fdist = FreqDist(words)

# Get hapaxes (Hapax legomenon), i.e. words that occur only once.
# Same as [w for w in words if words.count(w)==1]
hapaxes = [w for w in fdist if fdist[w]==1]
print(hapaxes[:10])

# Print how many hapaxes there are in the text.
print(len(hapaxes))

# Get list of words that do not apper in English vocabulary.
english_vocab = set(w.lower() for w in nltk.corpus.words.words())
weird_words = sorted(set(vocab) - english_vocab)

# Create meaningful words ('content') list by removing common 'stopwords'.
stopwords = nltk.corpus.stopwords.words('english')
content = [w for w in words if w not in stopwords]

def find_collocations(tokens, num=20):
    """
    Returns collocations derived from tokens, ignoring stopwords.
    Collocations are returned as a list of tuples.
    Parameter num: The maximum number of collocations to return
    """
    from nltk.collocations import BigramCollocationFinder
    from nltk.metrics import BigramAssocMeasures
    from nltk.corpus import stopwords

    ignored_words = nltk.corpus.stopwords.words('english')
    colloc_finder = BigramCollocationFinder.from_words(tokens)

    # Get only those bigrams that appear at least twice in text.
    colloc_finder.apply_freq_filter(2)

    # Get only those bigrams that inlcude words longer than 2 characters
    # and are not common english 'stopwords'.
    colloc_finder.apply_word_filter(lambda w: len(w) < 3 or w.lower() in ignored_words)
    collocations = colloc_finder.nbest(BigramAssocMeasures.likelihood_ratio, num)
    return collocations

# Get 40 most common collocations (words that frequuently appear together).
collocations = find_collocations(tokens, 40)

# Draw lexical dispersion plot for below words.
dispersion_plot(tokens, ['Poirot', 'Hastings', 'Cavendish'])
