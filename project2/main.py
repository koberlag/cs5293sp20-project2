import glob
import json
import os
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn import metrics
import spacy
import nltk
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.chunk import conlltags2tree, tree2conlltags
import unicodedata
import re
from contractions import CONTRACTION_MAP
from normalize import normalize_corpus, parse_document
from string import punctuation
import networkx


# Max number of files to read for clustering and summarizing
MAX_FILE_COUNT = 5000

# # Create spacy nlp object
NLP = spacy.load("en_core_web_sm")

#10, 20, 0.266
# Number of clusters to use in the clustering function
NUM_CLUSTERS = 6

# Number of features per cluster
TOP_N_FEATURES = 5

# Directory containing sample of json files for clustering and summarizing
DIR_TO_READ = "CORD-19-research-challenge/comm_use_subset/comm_use_subset/pdf_json/"

def get_json_file_paths():
    """Uses the 'DIR_TO_RUN' path and a *.json wildcard to create a glob
    for getting file paths to .json files in the given directory.
    A random sample up to either the 'MAX_FILE_COUNT' or the max number of json files
    available in the given directory will be returned."""

     # Get system directory from given relative path
    cord_comm_use_subset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), DIR_TO_READ)
    # Use directory and wildcard to return glob object of .json file paths
    json_glob = glob.glob(cord_comm_use_subset_dir + "*.json")
    # Set max file count to the global param, unless the glob contains fewer elements, in which case set to length of glob
    max_file_count = MAX_FILE_COUNT if len(json_glob) >= MAX_FILE_COUNT else len(json_glob)
    # Get a random sample of .json file paths
    json_paths = random.sample(json_glob, max_file_count)

    return json_paths

def get_text_data():
    """Extracts body text from each json file in the given 'DIR_TO_READ' directory.
    Returns a list of dictionaries with the file name text data for each file that is read."""

    # A list for holding the necessary data from each .json file
    document_list = []
    json_paths = get_json_file_paths()
    
    for json_path in json_paths:
        # Separate file name and file ext.
        file_path, file_ext = os.path.splitext(json_path)
        directory, file_name = os.path.split(file_path)
        try:
            # Read file by path
            with open(json_path, "r") as f:
                # read/parse json data
                data = json.load(f)
                
                if(data is not None):
                    # Create dict to hold file name and text
                    paragraph_dict = {'file_name': file_name + file_ext, 'text': []}
                    # for each paragraph in body_text
                    for body_text in data['body_text']:
                        # Add paragraph to text list in the dict
                        paragraph_dict['text'].append(body_text['text'])
                        
                    # Append file data to document_list
                    document_list.append(paragraph_dict)
                else:
                    print(f"No data found for {file_name}{file_ext}")
                    continue
        except Exception as ex:
            print(f"Could not read or extract data from {file_name}{file_ext}")
            print(ex)
    return document_list

def write_to_summary_file(file_name, summary_sentences):
    with(open("SUMMARY.md", "a")) as f:
        f.write(f"File Name: {file_name} \n")
        f.writelines(summary_sentences)
        f.write("\n\n")



from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def build_feature_matrix(documents, feature_type='frequency'):

    feature_type = feature_type.lower().strip()  
    
    if feature_type == 'binary':
        vectorizer = CountVectorizer(binary=True, min_df=1, 
                                     ngram_range=(1, 1))
    elif feature_type == 'frequency':
        vectorizer = CountVectorizer(binary=False, min_df=1, 
                                     ngram_range=(1, 1))
    elif feature_type == 'tfidf':
        vectorizer = TfidfVectorizer(min_df=1, 
                                     ngram_range=(1, 1))
    else:
        raise Exception("Wrong feature type entered. Possible values: 'binary', 'frequency', 'tfidf'")

    feature_matrix = vectorizer.fit_transform(documents).astype(float)
    
    return vectorizer, feature_matrix


from scipy.sparse.linalg import svds
    
def low_rank_svd(matrix, singular_count=2):
    
    u, s, vt = svds(matrix, k=singular_count)
    return u, s, vt

    

def textrank_text_summarizer(sentences,  num_sentences=2,
                             feature_type='frequency'):
    try:
        vec, dt_matrix = build_feature_matrix(sentences, 
                                      feature_type='tfidf')
        similarity_matrix = (dt_matrix * dt_matrix.T)
            
        similarity_graph = networkx.from_scipy_sparse_matrix(similarity_matrix)
        scores = networkx.pagerank(similarity_graph)   
        
        ranked_sentences = sorted(((score, index) 
                                    for index, score 
                                    in scores.items()), 
                                reverse=True)

        top_sentence_indices = [ranked_sentences[index][1] 
                                for index in range(num_sentences)]
        top_sentence_indices.sort()
        
        top_sentences = []
        for index in top_sentence_indices:
            top_sentences.append(sentences[index]) 
        return top_sentences     
    except Exception as ex:
        print(ex) 
        return []
   

def cluster_document(file_name, sentences):
    normalized_corpus = ' '.join(normalize_corpus(sentences))
    tfidf_vectorizer = TfidfVectorizer()

    pos_tag = ['PROPN', 'ADJ', 'NOUN', 'VERB']
    text_spacy = NLP(normalized_corpus)
    normalized_corpus = []
    for token in text_spacy:
        if(token.pos_ in pos_tag):
            word = token.lemma_ if token.lemma_ != '-PRON-' else token.text
            normalized_corpus.append(word)
    try:
        tfidf_matrix = tfidf_vectorizer.fit_transform(normalized_corpus)
        km = KMeans(n_clusters=NUM_CLUSTERS).fit(tfidf_matrix)
        # km = KMeans(n_clusters=NUM_CLUSTERS, max_iter=10000, n_init=50, random_state=42).fit(tfidf_matrix)
        feature_names = tfidf_vectorizer.get_feature_names()
        topn_features = TOP_N_FEATURES
        ordered_centroids = km.cluster_centers_.argsort()[:, ::-1]
        # print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(tfidf_matrix, km.labels_, sample_size=1000))
        
        # print('File Name: ' + file_name)
        # get key features for each cluster
        for cluster_num in range(NUM_CLUSTERS):
            key_features = [feature_names[index] for index in ordered_centroids[cluster_num, :topn_features]]
            # print('CLUSTER #'+str(cluster_num+1))
            # print('Key Features:', key_features)
            # print('-'*80)
    except Exception as ex:
        print(ex)
   



def main():
    document_data = get_text_data()
    
    # Remove summary file if exists
    if(os.path.isfile("SUMMARY.md")):
        os.remove('SUMMARY.md')
        with(open("SUMMARY.md", "a")) as f:
            f.writelines(["THIS IS A SUMMARY FILE OF THE CORD-19 FILE DATA.\n", 
                            "THE FOLLOWING SUMMARY INFORMATION WAS DETERMINED USING THE TEXT RANK ALGORITHM:\n",
                            "\n"])
        
    import time
    import sys
    time_start = time.time()
    seconds = 0
    minutes = 0
    f_count = 0
    for document in document_data:
        f_count += 1
        file_name = document['file_name']
        # List of paragraphs from file
        text = document['text']
        # Join list of paragraphs into a single string
        file_corpus = ' '.join(text)
        sentences = parse_document(file_corpus)
        # cluster_document(file_name, sentences)
        summary_sentences = textrank_text_summarizer(sentences)
        write_to_summary_file(file_name, summary_sentences)
        print(f" File number: {f_count}")
        try:
            sys.stdout.write("\r{minutes} Minutes {seconds} Seconds".format(minutes=minutes, seconds=seconds))
            sys.stdout.flush()
            time.sleep(1)
            seconds = int(time.time() - time_start) - minutes * 60
            if seconds >= 60:
                minutes += 1
                seconds = 0
        except KeyboardInterrupt as e:
            break
    # document_clusters = cluster_documents(document_data)
    # document_summaries = summarize_document_clusters(document_clusters)
  


if __name__ == "__main__":
    main()