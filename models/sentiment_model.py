"""Sentiment analysis model implementation."""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import torch
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class SentimentAnalyzer:
    """Class for sentiment analysis of social media posts."""
    
    def __init__(self, model_type="transformer", model_path=None):
        """
        Initialize the sentiment analyzer.
        
        Args:
            model_type (str): Type of model to use ('transformer', 'custom')
            model_path (str): Path to saved model if using custom model
        """
        self.model_type = model_type
        self.model_path = model_path
        self.model = None
        self.vectorizer = None
        self.stop_words = set(stopwords.words('english'))
        self.is_trained = False
        
        if model_type == "transformer":
            # Use pre-trained transformer model
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
                self.model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
                self.nlp = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)
                self.is_trained = True
            except Exception as e:
                print(f"Error loading transformer model: {e}")
                # Fallback to simpler model if transformer fails
                self.model_type = "fallback"
        elif model_type == "custom" and model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def preprocess_text(self, text):
        """
        Preprocess text for sentiment analysis.
        
        Args:
            text (str): Text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Tokenize
        word_tokens = word_tokenize(text)
        
        # Remove stopwords and non-alphanumeric
        filtered_text = [word for word in word_tokens if word not in self.stop_words and word.isalnum()]
        
        return " ".join(filtered_text)
    
    def train(self, texts, labels, model_algorithm="svm"):
        """Train a custom sentiment analysis model"""
        # Create TF-IDF features
        self.vectorizer = TfidfVectorizer(max_features=5000)
        X = self.vectorizer.fit_transform(texts)
        
        # Initialize the model
        if model_algorithm == "svm":
            self.model = SVC(kernel='linear', probability=True)
        else:  # random_forest
            self.model = RandomForestClassifier(n_estimators=100)
        
        # Train the model
        self.model.fit(X, labels)
        
        # Generate predictions for metrics
        predictions = self.model.predict(X)
        
        # Calculate metrics
        report = classification_report(labels, predictions, output_dict=True)
        
        self.model_type = "custom"
        self.is_trained = True
        
        return {
            'accuracy': report['accuracy'],
            'classification_report': report
        }
    
    def save_model(self, path):
        """Save the custom model and vectorizer"""
        if self.model_type == "custom" and self.model:
            model_data = {
                'model': self.model,
                'vectorizer': self.vectorizer
            }
            joblib.dump(model_data, path)
            self.model_path = path
            print(f"Model saved to {path}")
        else:
            print("Only custom models can be saved.")
    
    def load_model(self, path):
        """Load a saved custom model"""
        if os.path.exists(path):
            model_data = joblib.load(path)
            self.model = model_data['model']
            self.vectorizer = model_data['vectorizer']
    
    def predict(self, text):
        """Predict sentiment for a single text"""
        if self.model_type == "transformer":
            result = self.model(text)[0]
            return {
                'label': 'positive' if result['label'] == 'POSITIVE' else 'negative',
                'score': result['score']
            }
        elif self.model_type == "custom" and self.model:
            # Transform text using vectorizer
            X = self.vectorizer.transform([text])
            
            # Get prediction and probability
            prediction = self.model.predict(X)[0]
            prob = np.max(self.model.predict_proba(X)[0])
            
            return {
                'label': prediction,
                'score': prob
            }
        else:
            # Fallback to TextBlob
            blob = TextBlob(text)
            sentiment = blob.sentiment.polarity
            
            if sentiment > 0:
                return {'label': 'positive', 'score': sentiment}
            elif sentiment < 0:
                return {'label': 'negative', 'score': abs(sentiment)}
            return {'label': 'neutral', 'score': 0.5}
    
    def batch_predict(self, texts):
        """Predict sentiment for a batch of texts"""
        return [self.predict(text) for text in texts]