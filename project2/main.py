import glob
import json
import os
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn import metrics
import spacy
import nltk
# nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.chunk import conlltags2tree, tree2conlltags
import unicodedata
import re
from contractions import CONTRACTION_MAP
# from normalize import normalize_corpus, parse_document
from normalization import normalize_corpus
from string import punctuation
import networkx
import pandas as pd

# Max number of files to read for clustering and summarizing
MAX_FILE_COUNT = 10

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
    file_count = MAX_FILE_COUNT if len(json_glob) >= MAX_FILE_COUNT else len(json_glob)
    # Get a random sample of .json file paths
    random.seed(42)
    json_paths = random.sample(json_glob, file_count)

    return json_paths

def get_file_text_data_list():
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

# def build_feature_matrix(documents, feature_type='frequency'):

#     feature_type = feature_type.lower().strip()  
    
#     if feature_type == 'binary':
#         vectorizer = CountVectorizer(binary=True, min_df=1, 
#                                      ngram_range=(1, 1))
#     elif feature_type == 'frequency':
#         vectorizer = CountVectorizer(binary=False, min_df=1, 
#                                      ngram_range=(1, 1))
#     elif feature_type == 'tfidf':
#         vectorizer = TfidfVectorizer(min_df=1, 
#                                      ngram_range=(1, 1))
#     else:
#         raise Exception("Wrong feature type entered. Possible values: 'binary', 'frequency', 'tfidf'")

#     feature_matrix = vectorizer.fit_transform(documents).astype(float)
    
#     return vectorizer, feature_matrix

def build_feature_matrix(documents, feature_type='frequency',
                         ngram_range=(1, 1), min_df=0.0, max_df=1.0):

    feature_type = feature_type.lower().strip()  
    
    if feature_type == 'binary':
        vectorizer = CountVectorizer(binary=True, min_df=min_df,
                                     max_df=max_df, ngram_range=ngram_range)
    elif feature_type == 'frequency':
        vectorizer = CountVectorizer(binary=False, min_df=min_df,
                                     max_df=max_df, ngram_range=ngram_range)
    elif feature_type == 'tfidf':
        vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df, 
                                     ngram_range=ngram_range)
    else:
        raise Exception("Wrong feature type entered. Possible values: 'binary', 'frequency', 'tfidf'")

    feature_matrix = vectorizer.fit_transform(documents).astype(float)
    
    return vectorizer, feature_matrix


from scipy.sparse.linalg import svds
    
def low_rank_svd(matrix, singular_count=2):
    
    u, s, vt = svds(matrix, k=singular_count)
    return u, s, vt

    

def textrank_text_summarizer(sentences, dt_matrix, num_sentences=2,
                             feature_type='frequency'):
    try:
        # vec, dt_matrix = build_feature_matrix(sentences, 
        #                               feature_type='tfidf')
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

from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
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
        # km = KMeans(n_clusters=NUM_CLUSTERS).fit(tfidf_matrix)
        km = KMeans(n_clusters=NUM_CLUSTERS, max_iter=10000, n_init=50, random_state=42).fit(tfidf_matrix)
        feature_names = tfidf_vectorizer.get_feature_names()
        topn_features = TOP_N_FEATURES
        ordered_centroids = km.cluster_centers_.argsort()[:, ::-1]
        print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(tfidf_matrix, km.labels_, sample_size=1000))
        visualizer = KElbowVisualizer(KMeans(), k=(2,32), metric='silhouette', timings=False)
        visualizer = SilhouetteVisualizer(KMeans(13))
        visualizer.fit(tfidf_matrix)
        visualizer.poof()
        # Use the quick method and immediately show the figure
        # kelbow_visualizer(KMeans(random_state=4), tfidf_matrix, k=(2,10))

        print('File Name: ' + file_name)
        # get key features for each cluster
        for cluster_num in range(NUM_CLUSTERS):
            key_features = [feature_names[index] for index in ordered_centroids[cluster_num, :topn_features]]
            print('CLUSTER #'+str(cluster_num+1))
            print('Key Features:', key_features)
            print('-'*80)
    except Exception as ex:
        print(ex)
   

def remove_add_summary():
     if(os.path.isfile("SUMMARY.md")):
        os.remove('SUMMARY.md')
        with(open("SUMMARY.md", "a")) as f:
            f.writelines(["THIS IS A SUMMARY FILE OF THE CORD-19 FILE DATA.\n", 
                            "THE FOLLOWING SUMMARY INFORMATION WAS DETERMINED USING THE TEXT RANK ALGORITHM:\n",
                            "\n"])

def k_means(feature_matrix, num_clusters=5):
    km = KMeans(n_clusters=num_clusters,
                max_iter=10000)
    km.fit(feature_matrix)
    clusters = km.labels_
    return km, clusters

def get_cluster_data(clustering_obj, file_data, 
                     feature_names, num_clusters,
                     topn_features=10):

    cluster_details = {}  
    # get cluster centroids
    ordered_centroids = clustering_obj.cluster_centers_.argsort()[:, ::-1]
    # get key features for each cluster
    # get files belonging to each cluster
    for cluster_num in range(num_clusters):
        cluster_details[cluster_num] = {}
        cluster_details[cluster_num]['cluster_num'] = cluster_num
        key_features = [feature_names[index] 
                        for index in ordered_centroids[cluster_num, :topn_features]]
        cluster_details[cluster_num]['key_features'] = key_features
        
        files = file_data[file_data['Cluster'] == cluster_num]['file_name'].values.tolist()
        cluster_details[cluster_num]['files'] = files
    
    return cluster_details

def print_cluster_data(cluster_data):
    # print cluster details
    for cluster_num, cluster_details in cluster_data.items():
        print('Cluster {} details:'.format(cluster_num))
        print('-'*20)
        print('Key features:', cluster_details['key_features'])
        print('Files in this cluster:')
        print(', '.join(cluster_details['files']))
        print('='*40)

def main():
    
    # Remove summary file if exists and add an empty one with header
    remove_add_summary()

    # Get the text data split per file, in a list  
    document_data = get_file_text_data_list()

    file_data = pd.DataFrame(document_data)
    # join the paragraphs for each file into a single string... list is still split per file
    document_data_text = [(' '.join(doc['text'])) for doc in document_data]
    #normalize the documents
    # norm_corpus = normalize_corpus(document_data_text)

    norm_file_corpus = normalize_corpus(corpus=document_data_text, lemmatize=True, only_text_chars=True)
    # extract tf-idf features
    vectorizer, feature_matrix = build_feature_matrix(norm_file_corpus,
                                                feature_type='frequency',
                                                min_df=0.24, max_df=0.85,
                                                ngram_range=(1, 2))

    # get feature names
    feature_names = vectorizer.get_feature_names() 
    # print sample features
    print (feature_names[:20])

    num_clusters = 5    
    km_obj, clusters = k_means(feature_matrix=feature_matrix,
                            num_clusters=num_clusters)

    file_data['Cluster'] = clusters

    cluster_data =  get_cluster_data(clustering_obj=km_obj,
                                 file_data=file_data,
                                 feature_names=feature_names,
                                 num_clusters=num_clusters,
                                 topn_features=5)         

    print_cluster_data(cluster_data) 


    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(feature_matrix, clusters, sample_size=1000))

    # visualizer = SilhouetteVisualizer(KMeans(5))
    # visualizer.fit(cv_matrix)
    # visualizer.poof()
    # # stop_words = nltk.corpus.stopwords.words('english')
    # # tf = TfidfVectorizer(ngram_range=(1, 2), min_df=10, max_df=0.8)
    # # tfidf_matrix = tf.fit_transform(norm_corpus)
    # # tfidf_matrix.shape

    # # km = KMeans(n_clusters=NUM_CLUSTERS, max_iter=10000, n_init=50, random_state=42).fit(tfidf_matrix)
    # # print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(tfidf_matrix, km.labels_, sample_size=1000))
        

    # cv = CountVectorizer(ngram_range=(1, 2), min_df=10, max_df=0.8)
    # cv_matrix = cv.fit_transform(norm_corpus)
    # cv_matrix.shape

    # NUM_CLUSTERS = 5
    # km = KMeans(n_clusters=NUM_CLUSTERS, max_iter=10000, n_init=50, random_state=42).fit(cv_matrix)
    # # km
    # # KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=10000,
    # # n_clusters=6, n_init=50, n_jobs=None, precompute_distances='auto',
    # # random_state=42, tol=0.0001, verbose=0)
    # file_data['kmeans_cluster'] = km.labels_

    # # km = KMeans(n_clusters=NUM_CLUSTERS, max_iter=10000, n_init=50, random_state=42).fit(cv_matrix)
    # print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(cv_matrix, km.labels_, sample_size=1000))

    # from collections import Counter
    # Counter(km.labels_)
    # # Counter({2: 429, 1: 2832, 3: 539, 5: 238, 4: 706, 0: 56})

    # # visualizer = KElbowVisualizer(KMeans(), k=(2,64), timings=False)
    # visualizer = SilhouetteVisualizer(KMeans(5))
    # visualizer.fit(cv_matrix)
    # visualizer.poof()

    # file_clusters = (file_data[['file_name', 'kmeans_cluster']]
    #                     .sort_values(by=['kmeans_cluster'],
    #                             ascending=False)
    #                     .groupby('kmeans_cluster').head(20))
    # file_clusters = file_clusters.copy(deep=True)
    # feature_names = cv.get_feature_names()
    # topn_features = 15
    # ordered_centroids = km.cluster_centers_.argsort()[:, ::-1]
    # # get key features for each cluster
    # # get movies belonging to each cluster
    # for cluster_num in range(NUM_CLUSTERS):
    #     key_features = [feature_names[index] for index in ordered_centroids[cluster_num, :topn_features]]
    #     files = file_clusters[file_clusters['kmeans_cluster'] == cluster_num]['file_name'].values.tolist()
    #     print('CLUSTER #'+str(cluster_num+1))
    #     print('Key Features:', key_features)
    #     print('Files', files)
    #     print('-'*80)

    # import time
    # import sys
    # time_start = time.time()
    # seconds = 0
    # minutes = 0
    # f_count = 0


    # tfidf_vectorizer = TfidfVectorizer()

    # # pos_tag = ['PROPN', 'ADJ', 'NOUN', 'VERB']
    # # text_spacy = NLP(norm_corpus)
    # # norm_corpus = []
    # # for token in text_spacy:
    # #     if(token.pos_ in pos_tag):
    # #         word = token.lemma_ if token.lemma_ != '-PRON-' else token.text
    # #         norm_corpus.append(word)
    # try:
    #     tfidf_matrix = tfidf_vectorizer.fit_transform(norm_corpus)
    #     # km = KMeans(n_clusters=NUM_CLUSTERS).fit(tfidf_matrix)
    #     km = KMeans(n_clusters=NUM_CLUSTERS, max_iter=10000, n_init=50, random_state=42).fit(tfidf_matrix)
    #     feature_names = tfidf_vectorizer.get_feature_names()
    #     topn_features = TOP_N_FEATURES
    #     ordered_centroids = km.cluster_centers_.argsort()[:, ::-1]
    #     print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(tfidf_matrix, km.labels_, sample_size=1000))
    #     visualizer = KElbowVisualizer(KMeans(), k=(4,48), timings=False)
    #     # visualizer = SilhouetteVisualizer(KMeans(13))
    #     visualizer.fit(tfidf_matrix)
    #     visualizer.poof()
    #     # Use the quick method and immediately show the figure
    #     # kelbow_visualizer(KMeans(random_state=4), tfidf_matrix, k=(2,10))

    #     print('File Name: ' + file_name)
    #     # get key features for each cluster
    #     for cluster_num in range(NUM_CLUSTERS):
    #         key_features = [feature_names[index] for index in ordered_centroids[cluster_num, :topn_features]]
    #         print('CLUSTER #'+str(cluster_num+1))
    #         print('Key Features:', key_features)
    #         print('-'*80)
    # except Exception as ex:
    #     print(ex)






    # # # document represents a single file
    # # for document in document_data:
    # #     f_count += 1
    # #     file_name = document['file_name']
    # #     # List of paragraphs from file
    # #     text = document['text']
    # #     # Join list of paragraphs into a single string
    # #     file_corpus = ' '.join(text)
    # #     sentences = parse_document(file_corpus)
    # #     cluster_document(file_name, sentences)
    # #     # summary_sentences = textrank_text_summarizer(sentences)
    # #     # write_to_summary_file(file_name, summary_sentences)
    # #     print(f" File number: {f_count}")
    # #     try:
    # #         sys.stdout.write("\r{minutes} Minutes {seconds} Seconds".format(minutes=minutes, seconds=seconds))
    # #         sys.stdout.flush()
    # #         time.sleep(1)
    # #         seconds = int(time.time() - time_start) - minutes * 60
    # #         if seconds >= 60:
    # #             minutes += 1
    # #             seconds = 0
    # #     except KeyboardInterrupt as e:
    # #         break
    # # # document_clusters = cluster_documents(document_data)
    # # # document_summaries = summarize_document_clusters(document_clusters)
  


if __name__ == "__main__":
    main()