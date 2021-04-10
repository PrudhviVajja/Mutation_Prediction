"""
Helper functions
-------
This module contains functions to help in the NLP pipeline. Current functions included - 
- Clean text
- Preprocess text
- Model Evaluation
"""

# Import libraries

import re
import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import string
import re
import contractions
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import spacy
import unicodedata

from sklearn.metrics import accuracy_score, classification_report, hamming_loss

# For wordnet lemmatizer
nltk.download('wordnet')

# For nltk tokenizer
nltk.download('punkt')



# Global variables
lemmatizer = WordNetLemmatizer()

nlp = spacy.load("en_core_web_sm")

# Helper functions
def clean_text_basic(df, text_to_clean = "content", cleaned_col = 'cleaned_text'):
    """
    Uses helper "clean_text()"
    Takes input df and text field name, and returns df with a new cleaned text field.
    
    Parameters
    ----------
    df : Pandas DataFrame
            Dataframe with the text field to be cleaned
    text_to_clean : str
            The name of the text field to be cleaned
    
    Returns
    -------
    Pandas DataFrame : Dataframe with the new cleaned text field added  
    """
    
    print("Started text cleaning for field: ", text_to_clean)
    # Lower casing
  
    df[cleaned_col] = df[text_to_clean].map(lambda text_to_clean: clean_text(str(text_to_clean)))
    
    print("Finished text cleaning, new cleaned field added: ", cleaned_col)
    
    return df


def clean_text(text):
    """
    Function to clean text with basic steps - lower casing, dealing with contractions, remove html codes,
    strip whitespaces, social media cleaning (remove hashtags and URLS), remove punctuationns, using regular expressions.
 
    Parameters
    ----------
    text : str
            Text to be cleaned
    
    Returns
    -------
    text : str
            Cleaned text
    """
    # Lower casing
    text = text.lower()
    
    
    # Remove html codes
    text = re.sub(r"&amp;", " ", text)
    text = re.sub(r"&quot;", " ", text)
    text = re.sub(r"&#39;", " ", text)
    text = re.sub(r"&gt;", " ", text)
    text = re.sub(r"&lt;", " ", text)
    
    # Strips (removes) whitespaces
    text = text.strip(' ')
    
    ################ Social media cleaning ############
    
    # Remove hashtags (Regex @[A-Za-z0-9]+ represents mentions and #[A-Za-z0-9]+ represents hashtags. )
    # text = re.sub("(@[A-Za-z0-9]+)|(#[A-Za-z0-9]+)", " ", text)
    
    # Remove URLS (Regex \w+:\/\/\S+ matches all the URLs starting with http:// or https:// and replacing it with space.)
    text = re.sub("(\w+:\/\/\S+)", " ", text)
    text = re.sub(r'http\S+', ' ', text)
    
     # remove old style retweet text "RT"
    # text = re.sub(r'^RT[\s]+', '', text)
    # remove accents
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    # remove @users
    # text = re.sub(r'@[\w]*', '', text)
    # remove Reddit channel reference /r
    # text = re.sub(r'r/', '', text)
    
    # remove reddit username
    # text = re.sub(r'u/[\w]*', '', text)
    # remove '&gt;' like notations
    # text = re.sub('&\W*\w*\W*;', ' ', text)
    # remove hashtags
    # text = re.sub(r'#[\w]*', '', text)
    ###################################################
    
    # Dealing with contractions
    text = contractions.fix(text)
    
    text = re.sub(r"what\'s", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can\'t", "can not ", text)
    text = re.sub(r"n\'t", " not ", text)
    text = re.sub(r"\'t", " not", text )
    text = re.sub(r"i\'m", "i am ", text)
    text = re.sub(r"\'em'", " them ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    
    
    # Removes punctuations
    text = re.sub('['+string.punctuation+']', " ", text)
    
	# Removes non alphanumeric characters
    text = re.sub('\W', ' ', text)
    
    # Removes non alphabetical characters
    #text = re.sub('[^a-zA-Z]+', ' ', text)
    
    # Replaces all whitespaces by 1 whitespace
    text = re.sub('\s+', ' ', text)
    
    return text




def preprocess_cleaned_text(df, text_to_preprocess = "cleaned_text", lemmatize_flag = True):
    """
    Fucntion to preprocess cleaned text. Steps:
    - tokenize
    - lemmatize
    
    Parameters
    ----------
    df : Pandas DataFrame
            Dataframe with the text field to be cleaned
    text_to_preprocess : str
            The name of the text field to be preprocessed
    lemmatize_flag : bool
            Set flag to true to lemmatize (default = true)
    
    Returns
    -------
    Pandas DataFrame : Dataframe with the new preprocessed text field added  
    """
    
    print("Started text preprocessing for field: ", text_to_preprocess)
    
    # Tokenize
    df['cleaned_processed'] = df[text_to_preprocess].map(lambda text: nltk.word_tokenize(text))
    
    # Lemmatize
    if lemmatize_flag:
        df.cleaned_processed = df.cleaned_processed.apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
    
    # Join into sentence
    df.cleaned_processed = df.cleaned_processed.apply(lambda x: ' '.join(x))

    print("Finished text preprocessing, new field added: cleaned_processed")
    
    return df

def linguistic_feature_gen(df, col = "text"):
    """
    Fucntion to generate linguistic features to be used in classification tasks.
    
    Parameters
    ----------
    df : Pandas DataFrame
            Dataframe with the text field to be cleaned
    col : str
            The name of the text field to be used to produce linguistic features
    
    Returns
    -------
    Pandas DataFrame : Dataframe with the new linguistic features added  
    """
    
    print("Started linguistic feature generation on column:", col)
    
    columns_before = df.columns

    df['total_length'] = df[col].apply(len)
    df[df.total_length == 0] = 1
    df['capitals'] = df[col].apply(lambda comment: sum(1 for c in comment if c.isupper()))
    df['caps_vs_length'] = df.apply(lambda row: float(row['capitals'])/float(row['total_length']),axis=1)
    df['num_exclamation_marks'] = df[col].apply(lambda comment: comment.count('!'))
    df['num_question_marks'] = df[col].apply(lambda comment: comment.count('?'))
    df['num_punctuation'] = df[col].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation))) 
    df['num_symbols'] = df[col].apply(lambda comment: sum(comment.count(w) for w in '*&$%'))
    df['num_words'] = df[col].apply(lambda comment: len(comment.split()))
    df['num_unique_words'] = df[col].apply(lambda comment: len(set(w for w in comment.split())))
    df[df.num_words == 0] = 1
    df['words_vs_unique'] = df['num_unique_words'] / df['num_words']
    df['char_count'] = df[col].apply(lambda x: sum(len(word) for word in str(x).split(" ")))
    #df['sentence_count'] = df["text"].apply(lambda x: len(str(x).split(".")))
    df['avg_word_length'] = df['char_count'] / df['num_words']
    #df['avg_sentence_length'] = df['word_count'] / df['sentence_count']
    df['upper_case_word_count'] = df[col].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))
    #df['vader_sent'] = df["text"].apply(lambda x: sid.polarity_scores(x))
    #df['compound'] = df['vader_sent'].apply(lambda x:x['compound'])
    
    columns_after = list(set(df.columns) - set(columns_before))
    
    print("Finished linguistic feature generation, created following new columns: ", columns_after)
    
    return df, columns_after

def remove_ner(data, text_col = 'text', pipe_disable=["tagger", "parser"], n_threads = 2, batch_size = 1000):
    """Function to remove Named Entities from text to avoid prediction bias.
       Returns new fields with the detected NER text, labels and a "no NER text field"""
    
    tokens = []
    lemma = []
    pos = []
    ner_text = []
    ner_label = []
    ner_no_ent_text = []

    for doc in nlp.pipe(data[text_col].astype('unicode').values, batch_size=batch_size,
                            n_threads=n_threads, disable=pipe_disable):
        
        # Commented code is for speed up by disabling unnecessary modules
        #tokens.append([n.text for n in doc])

        if doc.is_nered:
            ner_text.append([n.text for n in doc.ents])
            ner_label.append([n.label_ for n in doc.ents])
            ner_no_ent_text.append(' '.join([n.text for n in doc if not n.ent_type_]))
        else:
            ner.append(None)

    #     if doc.is_tagged:
    #         pos.append([n.pos_ for n in doc])
    #     else: 
    #         pos.append(None)
    #     if doc.is_parsed:    
    #         lemma.append([n.lemma_ for n in doc])
    #     else:
    #         lemma.append(None)


    # test['text_tokens'] = tokens
    # test['text_lemma'] = lemma
    # test['text_pos'] = pos
    data['text_ner_text'] = ner_text
    data['text_ner_label'] = ner_label
    data['text_no_ner'] = ner_no_ent_text
    
    print('Finished Named Entity Removal, added columns: [text_ner_text, text_ner_label, text_no_ner]')
    return data

def model_evaluation_comparison(model_name, y_labels, predictions, target_names, comparison_df = pd.DataFrame()):
    """
    Function for multilabel model evaluation and comparison. 
    Takes the labels and predictions as input,
    and returns model evaluation metrics in a comparative format.
    
    Parameters
    ----------
    model_name : str
            Name of the model
    y_labels : 1d array-like
            Ground truth (correct) target values.
    predictions : str
            Estimated targets as returned by a classifier.
    target_names : list
            List of target names/label names.
    comparison_df : Pandas DataFrame
            A dataframe containing previous model eval results

    Returns
    -------
    Pandas DataFrame
            A dataframe containing current model eval results (and appended to previous 
            if comparison_df is provided)
    """
    
    print("Started model evaluation script")
    
    # Calculate accuracy and hamming loss
    acc = accuracy_score(y_labels, predictions)
    print('Test accuracy is {}'.format(acc))
    hl = hamming_loss(y_labels, predictions)
    print('Hamming loss is {}'.format(hl))

    
    # Calculate and print classification report
    print(classification_report(y_labels, predictions, target_names=target_names))
    
    # Convert classification report to dataframe
    cr = classification_report(y_labels, predictions, target_names=target_names, output_dict = True)
    eval_df = pd.DataFrame.from_dict(cr)
    
    # Return all metrics formatted as a row per model
    eval_df = eval_df.reset_index()
         
    eval_df = eval_df.reset_index(drop=True)
    
    # Add metrics for comparison
    eval_df['accuracy'] = acc
    eval_df['hamming_loss'] = hl
    eval_df['model_name'] = model_name
    
    # Sort rows
    comparison_df = comparison_df.append(eval_df).sort_values(by=['index', 'model_name']).reset_index(drop=True)
    
    # Order columns
    if 'micro avg' in comparison_df.columns:
        comparison_df = comparison_df[['model_name', 'index'] + target_names + ['micro avg', 'macro avg', 'weighted avg', 'samples avg','accuracy', 'hamming_loss']]
    else:
        comparison_df = comparison_df[['model_name', 'index'] + target_names + ['macro avg', 'accuracy', 'hamming_loss']]
    
    print("Finished model evaluation script")
    
    return comparison_df
    
