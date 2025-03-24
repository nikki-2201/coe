"""Text preprocessing module for social media content."""

import re
import string
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import emoji

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class TextProcessor:
    """Class for preprocessing social media text data."""
    
    def __init__(self, language='english'):
        """
        Initialize the text processor.
        
        Args:
            language (str): Language for stopwords
        """
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()
        
    def remove_urls(self, text):
        """Remove URLs from text."""
        if not isinstance(text, str):
            return ""
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)
    
    def remove_mentions(self, text):
        """Remove @mentions from text."""
        if not isinstance(text, str):
            return ""
        return re.sub(r'@\w+', '', text)
    
    def remove_hashtags(self, text, keep_text=True):
        """
        Remove hashtags from text.
        
        Args:
            text (str): Input text
            keep_text (bool): If True, keeps the text of the hashtag without the #
            
        Returns:
            str: Text with hashtags removed
        """
        if not isinstance(text, str):
            return ""
        if keep_text:
            return re.sub(r'#(\w+)', r'\1', text)
        else:
            return re.sub(r'#\w+', '', text)
    
    def remove_emoji(self, text):
        """Remove emojis from text."""
        if not isinstance(text, str):
            return ""
        return emoji.replace_emoji(text, replace='')
    
    def remove_punctuation(self, text):
        """Remove punctuation from text."""
        if not isinstance(text, str):
            return ""
        translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator)
    
    def remove_numbers(self, text):
        """Remove numbers from text."""
        if not isinstance(text, str):
            return ""
        return re.sub(r'\d+', '', text)
    
    def remove_extra_whitespace(self, text):
        """Remove extra whitespace from text."""
        if not isinstance(text, str):
            return ""
        return re.sub(r'\s+', ' ', text).strip()
    
    def remove_stopwords(self, text):
        """Remove stopwords from text."""
        if not isinstance(text, str):
            return ""
        word_tokens = word_tokenize(text)
        filtered_text = [word for word in word_tokens if word.lower() not in self.stop_words]
        return ' '.join(filtered_text)
    
    def lemmatize_text(self, text):
        """Lemmatize text to reduce words to their base form."""
        if not isinstance(text, str):
            return ""
        word_tokens = word_tokenize(text)
        lemmatized_text = [self.lemmatizer.lemmatize(word) for word in word_tokens]
        return ' '.join(lemmatized_text)
    
    def clean_text(self, text, remove_stop=True, lemmatize=True):
        """
        Apply full text cleaning pipeline.
        
        Args:
            text (str): Text to clean
            remove_stop (bool): Whether to remove stopwords
            lemmatize (bool): Whether to lemmatize text
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
            
        # Apply cleaning steps
        text = self.remove_urls(text)
        text = self.remove_mentions(text)
        text = self.remove_hashtags(text, keep_text=True)
        text = self.remove_emoji(text)
        text = self.remove_punctuation(text)
        text = self.remove_numbers(text)
        text = self.remove_extra_whitespace(text)
        
        # Optional processing
        if remove_stop:
            text = self.remove_stopwords(text)
        if lemmatize:
            text = self.lemmatize_text(text)
            
        return text.lower()
    
    def prepare_data_for_model(self, df, text_column='text', remove_stop=True, lemmatize=True):
        """
        Prepare a DataFrame for modeling.
        
        Args:
            df (pandas.DataFrame): DataFrame with text data
            text_column (str): Name of the column containing text
            remove_stop (bool): Whether to remove stopwords
            lemmatize (bool): Whether to lemmatize text
            
        Returns:
            pandas.DataFrame: DataFrame with processed text
        """
        # Create a copy to avoid modifying the original
        processed_df = df.copy()
        
        # Handle missing text
        processed_df[text_column] = processed_df[text_column].fillna("").astype(str)
        
        # Apply text cleaning
        processed_df['processed_text'] = processed_df[text_column].apply(
            lambda x: self.clean_text(x, remove_stop=remove_stop, lemmatize=lemmatize)
        )
        
        return processed_df
    
    def extract_features(self, df, text_column='processed_text'):
        """
        Extract basic text features from processed text.
        
        Args:
            df (pandas.DataFrame): DataFrame with processed text
            text_column (str): Column with processed text
            
        Returns:
            pandas.DataFrame: DataFrame with additional text features
        """
        # Create a copy to avoid modifying the original
        feature_df = df.copy()
        
        # Text length features
        feature_df['text_length'] = feature_df[text_column].apply(len)
        feature_df['word_count'] = feature_df[text_column].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
        
        # Calculating average word length
        feature_df['avg_word_length'] = feature_df[text_column].apply(
            lambda x: sum(len(word) for word in x.split()) / max(len(x.split()), 1) if isinstance(x, str) else 0
        )
        
        return feature_df