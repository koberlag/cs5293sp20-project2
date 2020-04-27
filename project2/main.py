import glob
import json
import os
import random
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn import metrics
import spacy
import nltk
import re
from contractions import CONTRACTION_MAP
from normalization import normalize_corpus, parse_document
import networkx
import pandas as pd
import time
import sys
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

# Download stopwords
nltk.download('stopwords')

# Timer for diagnostics
time_start = time.time()
seconds = 0
minutes = 0

# Max number of files to read for clustering and summarizing
MAX_FILE_COUNT = 50

# Number of files read
FILE_COUNT = 0

# Create spacy nlp object
NLP = spacy.load("en_core_web_sm")

# Number of clusters to use in the clustering function
NUM_CLUSTERS = 4

# Number of features per cluster
TOP_N_FEATURES = 5

# Directory containing sample of json files for clustering and summarizing
DIR_TO_READ = "CORD-19-research-challenge/comm_use_subset/comm_use_subset/pdf_json/"

def get_json_file_paths():
    """Uses the 'DIR_TO_RUN' path and a *.json wildcard to create a glob
    for getting file paths to .json files in the given directory.
    A random sample up to either the 'MAX_FILE_COUNT' or the max number of json files
    available in the given directory will be returned."""

    global FILE_COUNT
     # Get system directory from given relative path
    cord_comm_use_subset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), DIR_TO_READ)
    # Use directory and wildcard to return glob object of .json file paths
    json_glob = glob.glob(cord_comm_use_subset_dir + "*.json")
    # Set max file count to the global param, unless the glob contains fewer elements, in which case set to length of glob
    FILE_COUNT = MAX_FILE_COUNT if len(json_glob) >= MAX_FILE_COUNT else len(json_glob)
    # Get a random sample of .json file paths
    random.seed(42)
    json_paths = random.sample(json_glob, FILE_COUNT)

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
                    paragraph_dict = {'file_name': file_name + file_ext, 'file_text': [], 'title': data['metadata']['title']}
                    # for each paragraph in body_text
                    for body_text in data['body_text']:
                        # Add paragraph to text list in the dict
                        paragraph_dict['file_text'].append(body_text['text'])
                        
                    # Append file data to document_list
                    document_list.append(paragraph_dict)
                else:
                    print(f"No data found for {file_name}{file_ext}")
                    continue
        except Exception as ex:
            print(f"Could not read or extract data from {file_name}{file_ext}")
            print(ex)
    return document_list

def write_to_summary_file(summaries):
    '''Writes the given information the SUMMARY.md file, appending to the existing data.'''
   
    with(open("SUMMARY.md", "a")) as f:
        for file_summary in summaries:

            f.write(f"Cluster: {file_summary['cluster_number']} \n")
            f.write(f"File Name: {file_summary['file_name']} \n")
            f.write(f"Title: {file_summary['title']} \n")
            f.writelines(f"Summary: {file_summary['file_summary_sents']}")
            f.write("\n\n")

def build_feature_matrix(documents, feature_type='frequency',
                         ngram_range=(1, 1), min_df=0.0, max_df=1.0):
    '''Builds a document-term feature matrix using wieghts like TF_IDF or Bag of Words'''

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

    
def textrank_text_summarizer(sentences, num_sentences=2, feature_type='frequency'):
    """Builds a document-term feature matrix, and computes a document similarity matrix
    by multiplying the matrix by its transpose. These documents(sentences) are fed into
    the PageRank algorithm to obtain a score for each sentence. The senteces are ranked
    based on teh score and the top sentences are returned as the summarization"""
    
    try:
        vec, dt_matrix = build_feature_matrix(sentences, feature_type='tfidf')

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


def remove_add_summary():
    '''Deletes the existing SUMMARY.md file and creates a new one with the default header'''
    
    if(os.path.isfile("SUMMARY.md")):
        os.remove('SUMMARY.md')
        with(open("SUMMARY.md", "a")) as f:
            f.writelines(["THIS IS A SUMMARY FILE OF THE CORD-19 FILE DATA.\n", 
                            "THE FOLLOWING SUMMARY INFORMATION WAS DETERMINED USING THE TEXT RANK ALGORITHM ON A RANDOM SAMPLE OF 5000 FILES:\n",
                            "\n"])

def k_means(feature_matrix, num_clusters=5):
    '''Creates a k-means cluster model and fits the model
     with the given document-term feature matrix.
     Returns the model and the clusters (model labels)'''
    km = KMeans(n_clusters=num_clusters,
                max_iter=10000)
    km.fit(feature_matrix)
    clusters = km.labels_
    return km, clusters

def get_cluster_data(clustering_obj, file_data, 
                     feature_names, num_clusters,
                     topn_features=10):
    '''Creates and returns an object with the cluster number, key features
    and file data per cluster.'''
    cluster_details = {}  
    # get cluster centroids
    ordered_centroids = clustering_obj.cluster_centers_.argsort()[:, ::-1]
    # get key features for each cluster
    # get files belonging to each cluster
    for cluster_num in range(num_clusters):
        cluster_details[cluster_num] = {}
        cluster_details[cluster_num]['cluster_num'] = cluster_num
        key_features = [feature_names[index] for index in ordered_centroids[cluster_num, :topn_features]]
        cluster_details[cluster_num]['key_features'] = key_features
        file_names = file_data[file_data['cluster'] == cluster_num]['file_name'].values.tolist()
        cluster_details[cluster_num]['file_names'] = file_names
    return cluster_details

def print_cluster_data(cluster_data):
    '''Prints the cluster number, key features, and the file names of the files
    used in each cluster'''
    # print cluster details
    for cluster_num, cluster_details in cluster_data.items():
        print('Cluster {} details:'.format(cluster_num))
        print('-'*20)
        print('Key features:', cluster_details['key_features'])
        print('Files in this cluster:')
        print(', '.join(cluster_details['file_names']))
        print('='*40)
        
file_num = 0
def build_summary_list(file_data, cluster_data):
    '''Loops over each cluster and summarizes each file per cluster. 
    Returns a list of summary sentences per cluster, for all clusters'''

    global file_num
    summaries = []
    # For each cluster
    for cluster_num, cluster_details in cluster_data.items():
        # For each file in the current cluster
        for index, file_name in enumerate(cluster_details['file_names']):
            # Get the title from the current file
            title = file_data[file_data['file_name'] == file_name]['title'].values.tolist()[0]

            # Get the original file text from the current file
            file_text = file_data[file_data['file_name'] == file_name]['file_text'].values.tolist()[0]

            # Tokenize file text by sentence
            file_sent_tokens = parse_document(' '.join(file_text))

            file_num += 1
            print_elapsed_time(f"Summarize Cluster {cluster_num}, file {index + 1}, file_num {file_num}")
            # Get top summary sentences
            summary_sentences = ' ' .join(textrank_text_summarizer(file_sent_tokens))

            # Create dict to hold file name, title and text
            summary_dict = {'cluster_number': cluster_num, 'file_name': file_name, 'title': title, 'file_summary_sents': summary_sentences}
            
            # Add to summaries list
            summaries.append(summary_dict)
    return summaries

def print_elapsed_time(msg):
    '''Method for displaying elapsed time output, for debugging purposes'''

    global time_start, seconds, minutes
    try:
        print(f"\r{msg}: {minutes} Minutes {seconds} Seconds")
        # sys.stdout.write()
        # sys.stdout.flush()
        # time.sleep(1)
        seconds = int(time.time() - time_start) - minutes * 60
        if seconds >= 60:
            minutes += 1
            seconds = 0
    except Exception as e:
        print(e)

def main():

    print_elapsed_time("Begin Program")
    print(f"Number Of Clusters to use: {NUM_CLUSTERS}")
    print(f"Max File Count: {MAX_FILE_COUNT}")


    # Remove summary file if exists and add an empty one with header
    remove_add_summary()

    print_elapsed_time("Get File Data")

    #2. Choose documents & 3. Write a files reader
    # Get the text data split per file, in a list  
    document_data = get_file_text_data_list()

    print(f"Files Read: {FILE_COUNT}")

    file_data = pd.DataFrame(document_data)

    # join the paragraphs for each file into a single string... list is still split per file
    document_data_text = [(' '.join(doc['file_text'])) for doc in document_data]
   
    print_elapsed_time("Normalize Data")
    #normalize the documents
    norm_file_corpus = normalize_corpus(corpus=document_data_text)

    print_elapsed_time("Vectorize")
    # extract bag of words features
    vectorizer, feature_matrix = build_feature_matrix(norm_file_corpus,
                                                feature_type='frequency',
                                                min_df=0.01, max_df=0.85,
                                                ngram_range=(1, 2))

    # get feature names
    feature_names = vectorizer.get_feature_names() 

    print_elapsed_time("K-Means Model")

    # 4. Cluster documents
    # Get k-means model and the cluster object
    km_obj, clusters = k_means(feature_matrix=feature_matrix, num_clusters=NUM_CLUSTERS)

    # add cluster indexes to file_data data frame
    file_data['cluster'] = clusters

    print_elapsed_time("Get Cluster Data")
    # Get object with the cluster number, key features and file data per cluster
    cluster_data =  get_cluster_data(clustering_obj=km_obj,
                                 file_data=file_data,
                                 feature_names=feature_names,
                                 num_clusters=NUM_CLUSTERS,
                                 topn_features=TOP_N_FEATURES)         

    # print_cluster_data(cluster_data) 

    print_elapsed_time("Build Summaries")
    
    # 5. Summarize document clusters
    summaries = build_summary_list(file_data, cluster_data)

    # 6. Write Summarized clusters to a file
    write_to_summary_file(summaries) 


    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(feature_matrix, clusters, sample_size=1000))

    # print_elapsed_time("Create K-Elbow Visualizer:")
    # visualizer = KElbowVisualizer(KMeans(), k=(2,16), timings=False)
    # # visualizer = SilhouetteVisualizer(KMeans(2))
    # visualizer.fit(feature_matrix)
    # visualizer.show()

    print_elapsed_time("End Program")


if __name__ == "__main__":
    main()