import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import emoji

class TextProcessor:
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('wordnet')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
        
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text):
        """Clean and normalize text"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Convert emojis to text
        text = emoji.demojize(text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags but keep the text
        text = re.sub(r'#', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def remove_stopwords(self, text):
        """Remove stop words from text"""
        words = word_tokenize(text)
        return ' '.join([word for word in words if word not in self.stop_words])
    
    def lemmatize_text(self, text):
        """Lemmatize text"""
        words = word_tokenize(text)
        return ' '.join([self.lemmatizer.lemmatize(word) for word in words])
    
    def prepare_text(self, text, remove_stops=True, lemmatize=True):
        """Prepare text for analysis"""
        # Clean the text
        text = self.clean_text(text)
        
        # Remove stopwords if requested
        if remove_stops:
            text = self.remove_stopwords(text)
        
        # Lemmatize if requested
        if lemmatize:
            text = self.lemmatize_text(text)
        
        return text
    
    def prepare_data_for_model(self, data, text_column, remove_stop=True, lemmatize=True):
        """Prepare dataset for model training or prediction"""
        if text_column not in data.columns:
            raise ValueError(f"Text column '{text_column}' not found in data")
        
        # Process each text in the dataset
        data['processed_text'] = data[text_column].apply(
            lambda x: self.prepare_text(x, remove_stop, lemmatize)
        )
        
        return data 