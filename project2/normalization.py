from contractions import CONTRACTION_MAP
import re
import nltk
import string
from nltk.stem import WordNetLemmatizer
import unicodedata

def tokenize_text(text):
    '''Uses nltk to work tokenize and strip text.'''
    tokens = nltk.word_tokenize(text) 
    tokens = [token.strip() for token in tokens]
    return tokens

def expand_contractions(text, contraction_mapping):
    '''Uses the CONTRACTION_MAP dictionary to replace contractions
    with their expanded counter part words.'''
    
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text
    

def lemmatize_text(text):
    '''Word tokenizes the given text, then 
    uses WordNetLemmatizer to create lemmatized text.'''
    tokens = tokenize_text(text)
    lemmatizer = WordNetLemmatizer()
    lemma_text = ' '.join([lemmatizer.lemmatize(token) for token in tokens])
    return lemma_text
    
def remove_accented_chars(text):
    '''Replaces characters that contain accents, 
    with the same character having no accent.'''
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

def remove_special_characters(text, remove_digits=False):
    '''Removes characters that are not white-space, 
    or alpha-numeric. e.g. punctuation.'''
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text
    
def remove_stopwords(text):
    '''Uses nltk stopwords list to filter
    out stopwords from the given text.'''
    stopword_list = nltk.corpus.stopwords.words('english')
    tokens = tokenize_text(text)
    filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text


def normalize_corpus(corpus, contraction_expansion=True,
accented_char_removal=True, text_lower_case=True,
text_lemmatization=True, special_char_removal=True,
stopword_removal=True, remove_digits=True):
    '''Removes accented characters, expands contractions, 
    makes the text lower case, removes extra new lines and white space, 
    lemmatizes, removes special characters and\or digits, removes stopwords.'''
    normalized_corpus = []
    # normalize each document in the corpus
    for doc in corpus:
        # remove accented characters
        if accented_char_removal:
            doc = remove_accented_chars(doc)
        # expand contractions
        if contraction_expansion:
            doc = expand_contractions(doc, CONTRACTION_MAP)
        # lowercase the text
        if text_lower_case:
            doc = doc.lower()
        # remove extra newlines
            doc = re.sub(r'[\r|\n|\r\n]+', ' ',doc)
        # lemmatize text
        if text_lemmatization:
            doc = lemmatize_text(doc)
        # remove special characters and\or digits
        if special_char_removal:
        # insert spaces between special characters to isolate them
            special_char_pattern = re.compile(r'([{.(-)!}])')
            doc = special_char_pattern.sub(" \\1 ", doc)
            doc = remove_special_characters(doc, remove_digits=remove_digits)
        # remove extra whitespace
        doc = re.sub(' +', ' ', doc)
        # remove stopwords
        if stopword_removal:
            doc = remove_stopwords(doc)
        normalized_corpus.append(doc)
    return normalized_corpus

def parse_document(document):
    '''removes new lines, and uses nltk to sentence tokenize the given text.'''
    document = re.sub('\n', ' ', document)
    document = document.strip()
    sentences = nltk.sent_tokenize(document)
    sentences = [sentence.strip() for sentence in sentences]
    return sentences