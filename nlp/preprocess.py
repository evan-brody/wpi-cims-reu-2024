from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string


def preprocess_text(text):
    """
    Preprocesses the given text by performing the following steps:
    1. Lowercasing and tokenization
    2. Removing punctuation
    3. Removing action words
    4. Removing stopwords
    5. Lemmatization
    6. Standardizing the text by replacing abbreviations

    Args:
        text (str): The text to be preprocessed.

    Returns:
        list: A list of preprocessed tokens.

    """
    # Lowercasing and tokenization
    tokens = word_tokenize(text.lower())

    # Remove punctuation
    tokens = [token for token in tokens if token not in string.punctuation]

    # Remove action words
    action_words = ["replace", "repair"]
    tokens = [token for token in tokens if token not in action_words]

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Standardize the text
    abbreviation_map = {"cyl": "cylinder", "hyd": "hydraulic", "eng": "engine"}
    tokens = [abbreviation_map.get(token, token) for token in tokens]

    return tokens
