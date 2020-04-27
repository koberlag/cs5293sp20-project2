from contractions import CONTRACTION_MAP
import re
import nltk
import string
from nltk.stem import WordNetLemmatizer
import unicodedata

def tokenize_text(text):
    tokens = nltk.word_tokenize(text) 
    tokens = [token.strip() for token in tokens]
    return tokens

def expand_contractions(text, contraction_mapping):
    
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
    
    
def get_nltk_pos_tags(text):
    '''Uses NLTK to return the parts of speech (POS) from the given nltk tokens'''
    pos_tags = nltk.pos_tag(tokenize_text(text))
    return pos_tags

from nltk.corpus import wordnet as wn

# Annotate text tokens with POS tags
def nltk_tag_to_wordnet_tag(nltk_tag):
    '''Converts an nltk_tag pos to the corresponding wordnet character'''
    if nltk_tag.startswith('J'):
        return wn.ADJ
    elif nltk_tag.startswith('V'):
        return wn.VERB
    elif nltk_tag.startswith('N'):
        return wn.NOUN
    elif nltk_tag.startswith('R'):
        return wn.ADV
    else:          
        return None

def lemmatize_text(text):
    tokens = tokenize_text(text)
    lemmatizer = WordNetLemmatizer()
    lemma_text = ' '.join([lemmatizer.lemmatize(token) for token in tokens])
    return lemma_text
    
# def lemmatize_text(text):
#     '''Uses NLTK to lemmatize a given list of pos words and tags'''
#     pos_tagged_text = get_nltk_pos_tags(text)
#     lemmas = []
#     lemmatizer = WordNetLemmatizer()
#     for word, tag in pos_tagged_text:
#         wn_tag = nltk_tag_to_wordnet_tag(tag)
#         if(wn_tag):
#             lemma = lemmatizer.lemmatize(word, wn_tag)
#             lemmas.append(lemma)
#         else:
#             lemmas.append(word.lower())
#     lemmatized_text = ' '.join(lemmas)
#     return lemmatized_text
    
def remove_accented_chars(text):
    '''Replaces characters that contain accents, 
    with the same character having no accent'''
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text
    
def remove_stopwords(text):
    stopword_list = nltk.corpus.stopwords.words('english')
    tokens = tokenize_text(text)
    filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

def keep_text_characters(text):
    filtered_tokens = []
    tokens = tokenize_text(text)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

def normalize_corpus(corpus, contraction_expansion=True,
accented_char_removal=True, text_lower_case=True,
text_lemmatization=True, special_char_removal=True,
stopword_removal=True, remove_digits=True):

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
# def normalize_corpus(corpus, lemmatize=True, 
#                      only_text_chars=False,
#                      tokenize=False):
    
#     normalized_corpus = []    
#     for text in corpus:
#         text = expand_contractions(text, CONTRACTION_MAP)
#         if lemmatize:
#             text = lemmatize_text(text)
#         else:
#             text = text.lower()
#         text = remove_special_characters(text)
#         text = remove_stopwords(text)
#         if only_text_chars:
#             text = keep_text_characters(text)
        
#         if tokenize:
#             text = tokenize_text(text)
#             normalized_corpus.append(text)
#         else:
#             normalized_corpus.append(text)
            
#     return normalized_corpus


def parse_document(document):
    document = re.sub('\n', ' ', document)
    document = document.strip()
    sentences = nltk.sent_tokenize(document)
    sentences = [sentence.strip() for sentence in sentences]
    return sentences