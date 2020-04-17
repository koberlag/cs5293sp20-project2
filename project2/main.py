import glob
import json
import os
import random

# Max number of files to read for clustering and summarizing
MAX_FILE_COUNT = 5000

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


def main():
    document_data = get_text_data()
    x = 0


if __name__ == "__main__":
    main()