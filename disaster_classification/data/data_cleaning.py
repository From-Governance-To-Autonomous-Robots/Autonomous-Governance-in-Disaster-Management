import re
import os
import pandas as pd
# from spellchecker import SpellChecker
from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

class TextCleaning:
    
    def __init__(self):
        # self.spell = SpellChecker()
        self.stop_words = set(stopwords.words('english'))
        # self.lemmatizer = WordNetLemmatizer()
        self.contraction_dict = {
            "ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because", 
            "could've": "could have", "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
            "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
            "he'd": "he would", "he'll": "he will", "he's": "he is", "how'd": "how did",
            "how'd'y": "how do you", "how'll": "how will", "how's": "how is", "I'd": "I would",
            "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have", "I'm": "I am",
            "I've": "I have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have",
            "it'll": "it will", "it'll've": "it will have", "it's": "it is", "let's": "let us",
            "ma'am": "madam", "mayn't": "may not", "might've": "might have", "mightn't": "might not",
            "mightn't've": "might not have", "must've": "must have", "mustn't": "must not",
            "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
            "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have",
            "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
            "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have",
            "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have",
            "so've": "so have", "so's": "so as", "this's": "this is", "that'd": "that would",
            "that'd've": "that would have", "that's": "that is", "there'd": "there would",
            "there'd've": "there would have", "there's": "there is", "they'd": "they would",
            "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have",
            "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not",
            "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
            "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will",
            "what'll've": "what will have", "what're": "what are", "what's": "what is", "what've": "what have",
            "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
            "where've": "where have", "who'll": "who will", "who'll've": "who will have",
            "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have",
            "will've": "will have", "won't": "will not", "won't've": "will not have",
            "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have",
            "y'all": "you all", "y'all'd": "you all would", "y'all'd've": "you all would have",
            "y'all're": "you all are", "y'all've": "you all have", "you'd": "you would",
            "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
            "you're": "you are", "you've": "you have"
        }
        self.contractions_re = re.compile('|'.join(self.contraction_dict.keys()))
    
    def do_text_cleanup(self, text):
        text = self.replace_contractions(text)
        text = self.clean_numbers(text)
        text = self.clean_special_characters(text)
        text = self.clean_tweet_text(text)
        # text = self.spell_check(text)
        text = self.remove_stopwords(text)
        # text = self.lemmatize(text)
        return text
    
    def clean_special_characters(self, x):
        pattern = r'[^a-zA-Z0-9\s]'
        text = re.sub(pattern, '', x)
        return text

    def clean_tweet_text(self, text):
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'@\w+', '', text)  # Remove mentions
        text = re.sub(r'#\w+', '', text)  # Remove hashtags
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
        return text

    def clean_numbers(self, x):
        if bool(re.search(r'\d', x)):
            x = re.sub(r'\d{5,}', '#####', x)
            x = re.sub(r'\d{4}', '####', x)
            x = re.sub(r'\d{3}', '###', x)
            x = re.sub(r'\d{2}', '##', x)
        return x

    def replace_contractions(self, text):
        def replace(match):
            return self.contraction_dict[match.group(0)]
        return self.contractions_re.sub(replace, text)
    
    # def spell_check(self, text):
    #     corrected_text = []
    #     for word in text.split():
    #         corrected_word = self.spell.correction(word)
    #         if corrected_word is None:
    #             corrected_word = word  # If correction is None, keep the original word
    #         corrected_text.append(corrected_word)
    #     return ' '.join(corrected_text)
    
    def remove_stopwords(self, text):
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stop_words]
        return ' '.join(tokens)
    
    # def lemmatize(self, text):
    #     tokens = word_tokenize(text)
    #     tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
    #     return ' '.join(tokens)

# Ensure required NLTK data is downloaded
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
