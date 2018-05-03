# -*- coding: utf-8 -*-
"""
Created on Thu May  3 11:39:13 2018

@author: Dyn
"""

## text cleaning
import re 
import unicodedata
from contraction import mapper_contraction
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
import spacy

nlp = spacy.load('en')
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')


## strip HTML, this removes all the html formats and leave the content
def strip_html_tags(text):
    '''
    >>> strip_html_tags("<div, class='myclass'>my text.</div>")
    'my text.'
    '''
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text


## Removing accented characters
# https://docs.python.org/2/library/unicodedata.html
def remove_accented_chars(text):
    ## this function translate unicode to ascii code, then decode it into utf
    '''
    >>> remove_accented_chars(u'Málaga')
    'Malaga'
    '''
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

## Expanding contractions
def expand_contractions(text, mapper_contraction=mapper_contraction):
    ## this function expand the contracted words
    '''
    >>> expand_contractions("I don't think she's here")
    'I do not think she is here'
    '''
    contractions_pattern = re.compile('({})'.format('|'.join(mapper_contraction.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = mapper_contraction.get(match) \
                                   if mapper_contraction.get(match) \
                                    else mapper_contraction.get(match.lower()) 
        expanded_contraction = first_char+expanded_contraction[1:] ## this avoid switching the upper case
        return expanded_contraction
    
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


## Removing URL
def remove_url(text):
    return(''.join(re.sub("(http[s]:\/\/)?(www\.)?\S*[.][a-zA-Z]{2,4}(\/)?\S+(\/)?", "", text)))

## Removing tweets like text
def remove_tweet(text):
    return(''.join(re.sub("(@[\w]+)|([Rr][Tt])", " ", text)))

## Removing Special Characters (punctuations)
def remove_special_characters(text):
    text = re.sub('[^a-zA-Z0-9\s]', '', text)
    return text

# # Lemmatizing text
def lemmatize_text(text):
    '''
    >>> lemmatize_text("he has not been discovered")
    "he has not been discovered"
    '''
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text

## Removing Stopwords
def remove_stopwords(text, is_lower_case=False):
    '''
    >>> remove_stopwords("he has not been discovered")
    "not discovered"
    '''
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text


## Text_cleaning
def clean_text(corpus, html_stripping=True, contraction_expansion=True,
               accented_char_removal=True, text_lemmatization=True,
               special_char_removal=True, stopword_removal=True,
               url_removal=True, tweet_removal=True, text_lower_case=True):
    
    cleaned_corpus = []
    
    for doc in corpus:
        
        if html_stripping:
            doc = strip_html_tags(doc)
        
        if accented_char_removal:
            doc = remove_accented_chars(doc)
            
        if contraction_expansion:
            doc = expand_contractions(doc)
            
        if text_lower_case:
            doc = doc.lower()
            
        if url_removal:
            doc = remove_url(doc)
            
        if tweet_removal:
            doc = remove_tweet(doc)
            
        # remove extra newlines
        doc = re.sub(r'[\r|\n|\r\n]+', ' ',doc)
        # insert spaces between special characters to isolate them    
        special_char_pattern = re.compile(r'([{.(-)!}])')
        doc = special_char_pattern.sub(" \\1 ", doc)
        
        if text_lemmatization:
            doc = lemmatize_text(doc)
            
        if special_char_removal:
            doc = remove_special_characters(doc)  
            
        # remove extra whitespace
        doc = re.sub(' +', ' ', doc)
        
        if stopword_removal:
            doc = remove_stopwords(doc, is_lower_case=text_lower_case)
            
        cleaned_corpus.append(doc)
        
    return cleaned_corpus








