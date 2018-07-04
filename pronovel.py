import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk_sents, ne_chunk
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.draw import dispersion_plot


class NovelText:
    """NovelText is going to analyse and dissect mystery novel text.
    Explanations: ...
    """

    def __init__(self, raw):
        """Constructs a NovelText from text raw string."""
        self.raw = raw
        self.tokens = word_tokenize(raw)
        self.sentences = sent_tokenize(raw)

    @staticmethod
    def _reverse_to_raw(tokens):
        import re
        substrings = []
        raw = ' '.join(tokens)
        # Remove whitespace before punctutation marks.
        raw = re.sub(r"""\s               # whitespace
                         (                # start of group '\1':
                         [])}!?:;'\".,]   # any one punctuation mark
                         (?:\s|$)         # whitespace or end-of-string
                         )                # end of group '\1'""",
                     r"\1", raw, flags=re.VERBOSE)
        # Remove whitespace between opening parenthesis and next word.
        raw = re.sub(r"""([[({])          # any one of opening parenthesis
                         \s               # whitespace""",
                     r"\1", raw, flags=re.VERBOSE)
        return raw
            
    @classmethod
    def from_tokens(cls, tokens):
        """Constructs a NovelText from list of tokens (strings of words
        and punctuation marks).
        """
        return cls(cls._reverse_to_raw(tokens))
