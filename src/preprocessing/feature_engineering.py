"""Feature engineering module for sentiment analysis."""

import pandas as pd
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

# Download VADER lexicon for sentiment analysis
nltk.download('vader_lexicon', quiet=True)

class FeatureEngineer:
    """Feature engineering for sentiment analysis."""
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.sia = SentimentIntensityAnalyzer()
        self.tfidf = TfidfVectorizer(max_features=100, stop_words='english')
        
    def extract_basic_features(self, df, text_column='processed_text'):
        """
        Extract basic text features.
        
        Args:
            df (pandas.DataFrame): DataFrame with text data
            text_column (str): Column name containing text
            
        Returns:
            pandas.DataFrame: DataFrame with additional features
        """
        # Create a copy to avoid modifying the original
        feature_df = df.copy()
        
        # Text length features
        feature_df['text_length'] = feature_df[text_column].apply(lambda x: len(x) if isinstance(x, str) else 0)
        feature_df['word_count'] = feature_df[text_column].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
        
        # Word-level features
        feature_df['avg_word_length'] = feature_df[text_column].apply(
            lambda x: sum(len(word) for word in x.split()) / max(len(x.split()), 1) if isinstance(x, str) else 0
        )
        
        # Character-level features
        feature_df['uppercase_ratio'] = feature_df[text_column].apply(
            lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1) if isinstance(x, str) else 0
        )
        
        return feature_df
    
    def extract_sentiment_features(self, df, text_column):
        """
        Extract VADER sentiment features.
        
        Args:
            df (pandas.DataFrame): DataFrame with text data
            text_column (str): Column name containing text
            
        Returns:
            pandas.DataFrame: DataFrame with sentiment features
        """
        # Create a copy to avoid modifying the original
        feature_df = df.copy()
        
        # Apply VADER sentiment analysis
        def get_vader_scores(text):
            if not isinstance(text, str) or not text:
                return {'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0}
            return self.sia.polarity_scores(text)
        
        # Extract sentiment scores
        vader_scores = feature_df[text_column].apply(get_vader_scores)
        
        # Add scores as new columns
        feature_df['vader_negative'] = vader_scores.apply(lambda x: x['neg'])
        feature_df['vader_neutral'] = vader_scores.apply(lambda x: x['neu'])
        feature_df['vader_positive'] = vader_scores.apply(lambda x: x['pos'])
        feature_df['vader_compound'] = vader_scores.apply(lambda x: x['compound'])
        
        return feature_df
    
    def extract_social_media_features(self, df, text_column):
        """
        Extract social media specific features.
        
        Args:
            df (pandas.DataFrame): DataFrame with text data
            text_column (str): Column name containing text
            
        Returns:
            pandas.DataFrame: DataFrame with social media features
        """
        # Create a copy to avoid modifying the original
        feature_df = df.copy()
        
        # Count hashtags, mentions, and URLs
        feature_df['hashtag_count'] = feature_df[text_column].apply(
            lambda x: len(re.findall(r'#\w+', x)) if isinstance(x, str) else 0
        )
        
        feature_df['mention_count'] = feature_df[text_column].apply(
            lambda x: len(re.findall(r'@\w+', x)) if isinstance(x, str) else 0
        )
        
        feature_df['url_count'] = feature_df[text_column].apply(
            lambda x: len(re.findall(r'https?://\S+|www\.\S+', x)) if isinstance(x, str) else 0
        )
        
        # Check for emojis (simple regex approach)
        emoji_pattern = re.compile("["
                               "😀-🙏"  # Emoji ranges
                               "😂-🤣"
                               "👍-👎"
                               "]+", flags=re.UNICODE)
        
        feature_df['emoji_count'] = feature_df[text_column].apply(
            lambda x: len(emoji_pattern.findall(x)) if isinstance(x, str) else 0
        )
        
        return feature_df
    
    def extract_tfidf_features(self, df, text_column, n_components=10):
        """
        Extract TF-IDF features and reduce dimensionality.
        
        Args:
            df (pandas.DataFrame): DataFrame with text data
            text_column (str): Column name containing text
            n_components (int): Number of components for dimensionality reduction
            
        Returns:
            pandas.DataFrame: DataFrame with TF-IDF features
        """
        # Create a copy to avoid modifying the original
        feature_df = df.copy()
        
        # Ensure the text column contains strings
        texts = feature_df[text_column].fillna("").astype(str).tolist()
        
        # Fit and transform using TF-IDF
        tfidf_features = self.tfidf.fit_transform(texts)
        
        # Convert to DataFrame
        tfidf_df = pd.DataFrame(
            tfidf_features.toarray(),
            columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
        )
        
        # Combine with original DataFrame
        result_df = pd.concat([feature_df.reset_index(drop=True), tfidf_df], axis=1)
        
        return result_df
    
    def extract_all_features(self, df, text_column, include_tfidf=False):
        """
        Extract all features for sentiment analysis.
        
        Args:
            df (pandas.DataFrame): DataFrame with text data
            text_column (str): Column name containing text
            include_tfidf (bool): Whether to include TF-IDF features
            
        Returns:
            pandas.DataFrame: DataFrame with all extracted features
        """
        # Apply each feature extraction step
        result_df = self.extract_basic_features(df, text_column)
        result_df = self.extract_sentiment_features(result_df, text_column)
        result_df = self.extract_social_media_features(result_df, text_column)
        
        # Optionally extract TF-IDF features
        if include_tfidf:
            result_df = self.extract_tfidf_features(result_df, text_column)
        
        return result_df