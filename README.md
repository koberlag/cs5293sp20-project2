# CS5293SP20-PROJECT2
    AUTHOR: RANDAL KEVIN OBERLAG JR.

## APPLICATION DESCRIPTION

THE FUNCTION OF THIS APPLICATION IS TO READ A RANDOM SAMPLE OF FILES FROM A DATA SET, NORMALIZE THE DATA SET FOR USING A K-MEANS MACHINE LEARNING ALGORITHM, AND LASTLY TO SUMMARIZE THE TEXT FOUND IN THE SAMPLED FILES, GROUPED BY THEIR CLUSTERS. THIS APPLICATION RELIES ON A DATA SET AVAILABLE FROM THE ALLEN INSTITUTE FOR AI, THROUGH KAGGLE (HTTPS://WWW.KAGGLE.COM/ALLEN-INSTITUTE-FOR-AI/CORD-19-RESEARCH-CHALLENGE). THE DATASET CONTAINS FILES OBTAINED THROUGH A LARGE BODY OF RESEARCH ON COVID-19.

THE BULK OF THIS APPLICATION UTILIZED TEXT ANALYTICS TECHNIQUES FOUND IN THE BOOK 'TEXT ANALYTICS WITH PYTHON: A PRACTITIONER'S GUIDE TO NATURAL LANGUAGE PROCESSING. (SECOND EDITION)' BY DIPANJAN SARKAR. HIS TECHNIQUES FOR NORMALIZING, CLUSTERING, AND UTILIZING THE TEXTRANK ALGORITHM FOR SUMMARIZATION WERE HEAVILY RELIED UPON FOR COMPLETING THIS PROJECT. 

IN ORDER TO MEASURE THE QUALITY OF THE CLUSTERS BEING MADE BY THE APPLICATION, A COUPLE OF APPROACHES WERE USED. FIRST, A SILOHOUETTE COEFFICIENT WAS USED, AND WITH THIS PARTICULAR SET OF DATA, NEVER REACHED WHAT I BELIEVE TO BE A VERY HIGH VALUE. ON AVERAGE, THE APPLICATION WILL SCORE ABOUT .3. IN ADDITION, THE YELLOWBRICK PACKAGE WAS USED IN ORDER TO VISUALIZE A K-ELBOW PLOT. I TRIED VARIOUS RANGES OF CLUSTERS, AND NONE OF WHICH GAVE A VERY WELL DEFINED ELBOW. WHEN THE APPLICATION IS RUNNING A FULL SAMPLE OF 5000 FILES, IT TAKES A VERY LONG TIME, AND THEREFORE I WAS NOT ABLE TO TEST OUT SEVERAL VARIATIONS OF CLUSTERING WITH THE FULL DATA SET. IN THE END A CLUSTER SIZE OF 4 WAS CHOSEN.

## DISCUSSION OF DATA FORMAT

THE OUTPUT OF THE SUMMARIZER INCLUDES THE CLUSTER NUMBER THAT THE FILE WAS ASSOCIATED WITH, THE FILE NAME, THE TITLE OF THE PAPER FOUND IN THE FILE, AND THE TOP TWO RANKED SUMMARY SENTENCES PRODUCED BY THE TEXTRANK ALGORITHM. THE FINAL SUMMARY CONTAINS THIS INFORMATION FOR 5000 RANDOMLY SAMPLED FILES FROM THE COM_USE_SUBSET (PDF_JSON) FILES WITHIN THE CORD-19 DATASET. THE ORIGINAL DATASET WAS PARSED USING THE PYTHON JSON PACKAGE, AND CONVERTED INTO A PYTHON DICTIONARY. WITH THE PYTHON DICTIONARY, THE APPLICATION IS ABLE TO SELECTED THE DESIRED ATTRIBUTES FROM THE ORIGINAL DATASET. THE ATTRIBUTES USED, WERE THE TITLE, FOUND IN THE METADATA DICTIONARY, AND THE TEXT, FOUND IN THE BODY_TEXT DICTIONARY. THE TEXT ATTRIBUTE CONTAINS A SET OF ALL THE PARAGRAPHS FOUND IN THE PAPER.

## DISCUSSION OF TOKENIZER

THERE WERE TWO TOKENIZATION TYPES THAT WERE USED IN THE PROJECT, BOTH OF WHICH USE THE NLTK LIBRARY FOR TOKENIZATION. FIRST, THE FULL TEXT PER FILE WAS NORMALIZED, AND IN ORDER TO ACCOMPLISH SOME OF THE DATA CLEANING, WORD TOKENIZATION WAS USED. IN ADDITION, SENTENCE TOKENIZATION WAS LATER USED FOR THE SENTENCE SUMMARIZATION FUNCTIONALITY. THE SENTENCE TOKENS WERE PASSED INTO THE TEXTRANK SUMMARIZER, FOR EACH FILE THAT WAS READ.

## DISCUSSION OF CLUSTERING METHOD

THE K-MEANS CLUSTERING METHOD WAS USED FOR THIS PROJECT, WITH A CLUSTER SIZE OF 4. AS MENTIONED ABOVE,THE SCORING MEASURES THAT WERE USED TO DETERMINE THE QUALITY OF CLUSTERS WERE NOT VERY CLEAR, AND THEREFORE IT SEEMED THAT SEVERAL DIFFERENT CLUSTER SIZES COULD HAVE BEEN CHOSEN WITH SIMILAR QUALITY RESULTS. I INITIALLY IMPLEMENTED THE K-MEANS CLUSTERING USING A TFIDF VECTORIZER FEATURE MATRIX, BUT AFTER TESTING OUT THE COUNT VECTORIZER, I FOUND THAT A BAG OF WORDS MATRIX PROVIDED A HIGHER SCORING METRIC.

## DISCUSSION OF SUMMARIZATION

THE SUMMARIZATION FUNCTIONALITY USES THE PARSED SENTENCE TOKENS OF THE ORIGINAL TEXT FILES AND INDIVIDUALLY SUMMARIZES THEM, BY RETURNING THE TOP TWO RANKED AND SORTED SENTENCES PRODUCED BY THE NETWORKX PAGERANK ALGORITHM. EACH FILE IS SUMMARIZED AND GROUPED ACCORDING TO THE CLUSTER THAT IT WAS FOUND TO BE IN, FROM THE CLUSTERING METHOD RUN PRIOR. THE PAGERANK ALGORITHM USES A SIMILARITY GRAPH PRODUCED BY THE FEATURE MATRIX AND ITS TRANSPOSE IN ORDER TO DERIVE SCORES FOR RANKING. THE SENTENCES ARE WRITTEN TO THE SUMMARY.MD FILE DESCRIBED IN THE DISCUSSION OF DATA FORMAT SECTION ABOVE.


## DOWNLOADING THE APPLICATION

USE GIT CLONE TO DOWNLOAD THIS APPLICATION. IN YOUR TERMINAL, NAVIGATE TO THE DIRECTORY THAT YOU WOULD LIKE THE APPLICATION TO RUN, THEN RUN THE FOLLOWING GIT COMMAND:

    GIT CLONE HTTPS://GITHUB.COM/KOBERLAG/CS5293SP20-PROJECT2.GIT

## INSTALLING THE APPLICATION

THIS APPLICATION RELIES ON THE PIPENV PACKAGE, SO YOU WILL NEED TO INSTALL PIPENV USING THE PIP PACKAGE MANAGEMENT TOOL. RUN THE FOLLOWING COMMAND TO INSTALL PIPENV: 

    PIP INSTALL PIPENV

ONCE PIPENV IS INSTALLED, YOU WILL THEN BE ABLE TO INSTALL THE REQUIRED PACKAGES THAT ARE USED BY THIS APPLICATION (EXTERNAL LIBRARIES LISTED BELOW). RUN THE FOLLOWING COMMAND TO INSTALL THE EXTERNAL LIBRARIES:

    PIPENV INSTALL

THIS APPLICATION ALSO RELIES ON A LANGUAGE PACKAGE FOR THE SPACY LIBRARY, THEREFORE THE FOLLOWING COMMAND MUST BE RUN FROM WITHIN YOUR TERMINAL (INSIDE YOUR VIRTUAL ENVIRONMENT):

    PYTHON -M SPACY DOWNLOAD EN_CORE_WEB_SM

## RUNNING THE APPLICATION

THE APPLICATION MAY BE RUN USING THE FOLLOWING COMMAND: 
    
    PIPENV RUN PYTHON PROJECT1/MAIN.PY


## APPLICATION FUNCTIONS:

- GET_JSON_FILE_PATHS: USES THE 'DIR_TO_RUN' PATH AND A *.JSON WILDCARD TO CREATE A GLOB
    FOR GETTING FILE PATHS TO .JSON FILES IN THE GIVEN DIRECTORY.
    A RANDOM SAMPLE UP TO EITHER THE 'MAX_FILE_COUNT' OR THE MAX NUMBER OF JSON FILES
    AVAILABLE IN THE GIVEN DIRECTORY WILL BE RETURNED.


- GET_FILE_TEXT_DATA_LIST: EXTRACTS BODY TEXT FROM EACH JSON FILE IN THE GIVEN 'DIR_TO_READ' DIRECTORY.
    RETURNS A LIST OF DICTIONARIES WITH THE FILE NAME TEXT DATA FOR EACH FILE THAT IS READ.


- WRITE_TO_SUMMARY_FILE: WRITES THE GIVEN INFORMATION TO THE SUMMARY.MD FILE, APPENDING TO THE EXISTING DATA.
   

- BUILD_FEATURE_MATRIX: BUILDS A DOCUMENT-TERM FEATURE MATRIX USING WIEGHTS LIKE TF_IDF OR BAG OF WORDS.

    
- TEXTRANK_TEXT_SUMMARIZER: BUILDS A DOCUMENT-TERM FEATURE MATRIX, AND COMPUTES A DOCUMENT SIMILARITY MATRIX
    BY MULTIPLYING THE MATRIX BY ITS TRANSPOSE. THESE DOCUMENTS(SENTENCES) ARE FED INTO
    THE PAGERANK ALGORITHM TO OBTAIN A SCORE FOR EACH SENTENCE. THE SENTECES ARE RANKED
    BASED ON TEH SCORE AND THE TOP SENTENCES ARE RETURNED AS THE SUMMARIZATION.
  
   
- REMOVE_ADD_SUMMARY: DELETES THE EXISTING SUMMARY.MD FILE AND CREATES A NEW ONE WITH THE DEFAULT HEADER
  
- K_MEANS: CREATES A K-MEANS CLUSTER MODEL AND FITS THE MODEL
    WITH THE GIVEN DOCUMENT-TERM FEATURE MATRIX.
    RETURNS THE MODEL AND THE CLUSTERS (MODEL LABELS).
    

- GET_CLUSTER_DATA: CREATES AND RETURNS AN OBJECT WITH THE CLUSTER NUMBER, KEY FEATURES
    AND FILE DATA PER CLUSTER.
   

- PRINT_CLUSTER_DATA:PRINTS THE CLUSTER NUMBER, KEY FEATURES, AND THE FILE NAMES OF THE FILES
    USED IN EACH CLUSTER.
    

- BUILD_SUMMARY_LIST: LOOPS OVER EACH CLUSTER AND SUMMARIZES EACH FILE PER CLUSTER. 
    RETURNS A LIST OF SUMMARY SENTENCES PER CLUSTER, FOR ALL CLUSTERS
    

- PRINT_ELAPSED_TIME: METHOD FOR DISPLAYING ELAPSED TIME OUTPUT, FOR DEBUGGING PURPOSES

- MAIN: MAIN ENTRY METHOD FOR THE APPLICATION. USES THE ABOVE METHODS TO READ, TOKENIZE, NORMALIZE, CLUSTER, AND SUMMARIZE FILES.


- TOKENIZE_TEXT: USES NLTK TO WORK TOKENIZE AND STRIP TEXT.

- EXPAND_CONTRACTIONS: USES THE CONTRACTION_MAP DICTIONARY TO REPLACE 
    CONTRACTIONS WITH THEIR EXPANDED COUNTER PART WORDS.


- LEMMATIZE_TEXT:
    WORD TOKENIZES THE GIVEN TEXT, THEN 
    USES WORDNETLEMMATIZER TO CREATE LEMMATIZED TEXT.
    

- REMOVE_ACCENTED_CHARS: REPLACES CHARACTERS THAT CONTAIN ACCENTS, 
    WITH THE SAME CHARACTER HAVING NO ACCENT.


- REMOVE_SPECIAL_CHARACTERS: REMOVES CHARACTERS THAT ARE NOT WHITE-SPACE, 
    OR ALPHA-NUMERIC. E.G. PUNCTUATION.
    
    
- REMOVE_STOPWORDS: USES NLTK STOPWORDS LIST TO FILTER
    OUT STOPWORDS FROM THE GIVEN TEXT.


- NORMALIZE_CORPUS: REMOVES ACCENTED CHARACTERS, EXPANDS CONTRACTIONS, 
    MAKES THE TEXT LOWER CASE, REMOVES EXTRA NEW LINES AND WHITE SPACE, 
    LEMMATIZES, REMOVES SPECIAL CHARACTERS AND\OR DIGITS, REMOVES STOPWORDS.


- PARSE_DOCUMENT: REMOVES NEW LINES, AND USES NLTK TO SENTENCE TOKENIZE THE GIVEN TEXT.


## KNOWN BUGS:

- NONE

## ADDITIONAL ASSUMPTIONS:

- NONE

## EXTERNAL LIBRARIES
-   SPACY
-   NLTK
-   SKLEARN
-   NUMPY
-   NETWORKX
-   MATPLOTLIB
-   YELLOWBRICK
-   PANDAS
-   PYLINT

## REFERENCES
-   TEXT ANALYTICS WITH PYTHON: A PRACTITIONER'S GUIDE TO NATURAL LANGUAGE PROCESSING. (SECOND EDITION). DIPANJAN SARKAR 

## EXTERNAL RESOURCES

- HTTPS://SCIKIT-LEARN.ORG/STABLE/MODULES/GENERATED/SKLEARN.METRICS.SILHOUETTE_SCORE.HTML -- FOR SILHOETTE SCORE TESTING