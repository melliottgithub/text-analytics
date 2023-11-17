import string
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer

def create_alphanum_tokenizer():
    return RegexpTokenizer(r"[a-zA-Z0-9-]+").tokenize

class WordTokenizer:
    def __init__(self, tokenizer=None, stopwords=None, lemmatizer=None, lowercase=True, min_length=2, max_length=15, remove_punctuation=True, remove_digits=True):
        self.tokenizer = word_tokenize if tokenizer is None else tokenizer
        self.stopwords = set(nltk.corpus.stopwords.words('english')) if stopwords is None else stopwords
        self.lemmatizer = lemmatizer
        self.lowercase = lowercase
        self.min_length = min_length
        self.max_length = max_length
        self.remove_punctuation = remove_punctuation
        self.remove_digits = remove_digits

    def tokenize(self, text):
        # Optionally convert to lowercase
        if self.lowercase:
            text = text.lower()

        # Tokenize
        tokens = self.tokenizer(text)

        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stopwords]

        # Remove punctuation
        if self.remove_punctuation:
            tokens = [token for token in tokens if token not in string.punctuation]

        # Remove digits
        if self.remove_digits:
            tokens = [token for token in tokens if not token.isdigit()]
            
        # Optionally lemmatize
        if self.lemmatizer is not None:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]

        # Remove stopwords and apply length constraints
        tokens = [token for token in tokens if self.min_length <= len(token) <= self.max_length]

        return tokens
