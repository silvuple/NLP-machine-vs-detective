from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk_sents, ne_chunk
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.draw import dispersion_plot
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.corpus import words, stopwords


class NovelText:
    """NovelText class for analysing a text of a mystery novel.

        Attributes:
        raw: A string containing the entire text of the novel.
        tokens: A list of strings, raw text devided to a list of words
                and punctuation marks.
        sentences: A list of strings, raw text devided to sentences.
    """

    def __init__(self, raw):
        """Constructs a NovelText from text raw string."""
        self.raw = raw
        self.tokens = word_tokenize(raw)
        self.sentences = sent_tokenize(raw)


    def get_words(self):
        """Gets only words (tokens that start with an alphanumeric character)
        from tokens by filtering out all the puntruation marks.
        """
        self.words = [w.lower() for w in self.tokens if w[0].isalnum()]
        return self.words


    def get_vocab(self):
        """Returns vocabulary of the text by geting all the unique words."""
        self.vocab = sorted(set(self.words))
        return self.vocab

        
    @staticmethod
    def _reverse_to_raw(tokens):
        import re
        substrings = []
        raw = ' '.join(tokens)
        # Remove whitespace before punctutation marks.
        raw = re.sub(r"""\s            # whitespace
                         (             # start of group '\1':
                         [])}!?:;.,]   # any one punctuation mark
                         (?:\s|$)      # whitespace or end-of-string
                         )             # end of group '\1'""",
                     r"\1", raw, flags=re.VERBOSE)
        # Remove whitespace between opening parenthesis and next word.
        raw = re.sub(r"""([[({])       # any one of opening parenthesis
                         \s            # whitespace""",
                     r"\1", raw, flags=re.VERBOSE)
        return raw


    @classmethod
    def from_tokens(cls, tokens):
        """Constructs a NovelText from list of tokens (list of strings
        of words and punctuation marks).
        """
        return cls(cls._reverse_to_raw(tokens))


    def find_collocations(self, num=20):
        """Returns collocations derived from tokens, ignoring stopwords.
        Collocations are returned as a list of tuples.
        Parameter num: The maximum number of collocations to return
        """
        ignored_words = stopwords.words('english')
        finder = BigramCollocationFinder.from_words(self.tokens)
        # Get only those bigrams that appear at least twice in text.
        finder.apply_freq_filter(2)
        # Get only those bigrams that inlcude words longer than 2 characters
        # and are not common english 'stopwords'.
        finder.apply_word_filter(lambda w: len(w) < 3 or w.lower() in ignored_words)
        collocations = finder.nbest(BigramAssocMeasures.likelihood_ratio, num)
        return collocations


    def get_persons(self):
        """Returns a sorted list of all the characteres (person names)
        which appear at least twice in the the text.
        """
        # Create tokenized sentences list (list of lists of lists).
        sents = [word_tokenize(sent) for sent in self.sentences]
        # Add Part-of-Speach tagging to tokenized sentences.
        sents = [pos_tag(sent) for sent in sents]
        # Get named entities from tagged sentences.
        self.chunked_sents = ne_chunk_sents(sents)
        # Get all named entities with label 'PERSON'.
        self.all_persons = []
        for tree in self.chunked_sents:
            for chunk in tree:
                if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
                    self.all_persons.append(' '.join(w[0] for w in chunk))
        # Remove named entities that appear only once in text.
        self.all_persons = [person for person in self.all_persons
                        if self.all_persons.count(person) > 1]
        # Get unqiue persons sorted alphabetically.
        self.persons = sorted(set(self.all_persons))            
        return self.persons


    def get_persons_count(self):
        """Returns a dictionary with count of every person in the list."""
        self.persons_count = {person: self.all_persons.count(person) for
                              person in self.all_persons}
        return(self.persons_count)


    def get_name_count(self, name):
        """Returns integer number of times a name appears in the raw text."""
        return self.raw.count(name)



