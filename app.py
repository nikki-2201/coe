import os
import sys
import time
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
from nltk.tokenize import word_tokenize
import io

# Add the parent directory to system path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from config import *
from models.sentiment_model import SentimentAnalyzer
from models.utils import (
    load_dataset, prepare_dataset_for_analysis, 
    detect_text_column, detect_datetime_column,
    sample_large_dataset, get_file_info
)
from src.preprocessing.text_processor import TextProcessor
from src.visualization.sentiment_plots import (
    create_sentiment_distribution_chart,
    create_sentiment_over_time_chart,
    create_sentiment_wordcloud
)
from src.visualization.trend_analysis import (
    detect_sentiment_trends,
    create_trend_chart
)
from src.dashboard import (
    show_dataset_info,
    show_sentiment_analysis_section,
    show_trend_analysis_section
)

# Set page configuration
st.set_page_config(
    page_title="Social Media Sentiment Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'sentiment_results' not in st.session_state:
    st.session_state.sentiment_results = None
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None
if 'processor' not in st.session_state:
    st.session_state.processor = None
if 'text_column' not in st.session_state:
    st.session_state.text_column = None
if 'time_column' not in st.session_state:
    st.session_state.time_column = None
if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = None
if 'realtime_mode' not in st.session_state:
    st.session_state.realtime_mode = False

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Main application
def main():
    # Title and description
    st.title("📊 Social Media Post Sentiment Analysis")
    st.write("Upload your social media data and analyze the sentiment of posts!")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # File upload
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        # Model selection
        model_type = st.selectbox(
            "Select model type",
            options=["transformer", "custom"],
            index=0,
            help="Transformer: Pre-trained model for better accuracy. Custom: Train your own model."
        )
        
        # Real-time simulation toggle
        realtime_simulation = st.checkbox(
            "Enable real-time simulation",
            value=False,
            help="Simulate real-time data updates."
        )
        
        if realtime_simulation:
            refresh_interval = st.slider(
                "Refresh interval (seconds)",
                min_value=5,
                max_value=60,
                value=10
            )
            st.session_state.realtime_mode = True
        else:
            st.session_state.realtime_mode = False
            
        # Text processing options
        st.subheader("Text Processing")
        remove_stopwords = st.checkbox("Remove Stopwords", value=True)
        lemmatize = st.checkbox("Lemmatize Text", value=True)
        
        # Training options (if custom model selected)
        if model_type == "custom":
            st.subheader("Model Training")
            train_model = st.checkbox("Train Custom Model", value=False)
            if train_model:
                train_model_algorithm = st.selectbox(
                    "Algorithm",
                    options=["svm", "random_forest"],
                    index=0
                )
        else:
            train_model = False
            train_model_algorithm = "svm"
            
        # About section
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This application analyzes sentiment in social media data.
        It can process historical data or simulate real-time analysis.
        """)

    # Main content
    if uploaded_file is not None:
        try:
            # Read the data
            df = pd.read_csv(uploaded_file)
            
            # Display raw data
            st.subheader("📋 Raw Data Preview")
            st.dataframe(df.head())
            
            # Select text column
            text_column = st.selectbox(
                "Select the column containing the text data:",
                df.columns
            )
            
            if st.button("Analyze Sentiment"):
                with st.spinner("Analyzing sentiments..."):
                    # Perform sentiment analysis
                    df['sentiment'] = df[text_column].apply(analyze_sentiment)
                    
                    # Create two columns for visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Sentiment distribution
                        st.subheader("📊 Sentiment Distribution")
                        sentiment_counts = df['sentiment'].value_counts()
                        fig = px.pie(values=sentiment_counts.values, 
                                   names=sentiment_counts.index,
                                   title="Sentiment Distribution",
                                   color_discrete_sequence=px.colors.qualitative.Set3)
                        st.plotly_chart(fig)
                    
                    with col2:
                        # Word Cloud
                        st.subheader("☁️ Word Cloud")
                        text_data = " ".join(df[text_column].astype(str))
                        wordcloud = generate_wordcloud(text_data)
                        
                        # Convert the plot to an image
                        fig, ax = plt.subplots()
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis('off')
                        st.pyplot(fig)
                    
                    # Display results table
                    st.subheader("📑 Detailed Analysis Results")
                    st.dataframe(df[[text_column, 'sentiment']])
                    
                    # Download results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Results",
                        data=csv,
                        file_name="sentiment_analysis_results.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Please make sure your CSV file is properly formatted and contains text data.")
    else:
        # No file uploaded yet
        st.info("Please upload a file to begin analysis.")
        
        # Show sample dataset option
        if st.button("Use Sample Dataset"):
            sample_file_path = os.path.join(SAMPLE_DIR, "sample_twitter_data.csv")
            
            # Check if sample file exists, otherwise create it
            if not os.path.exists(sample_file_path):
                create_sample_dataset(sample_file_path)
                
            load_data(sample_file_path)
            st.experimental_rerun()

def load_data(file_path):
    """Load data from file path and prepare it for analysis."""
    with st.spinner("Loading data..."):
        # Load the dataset
        df = load_dataset(file_path)
        
        if not df.empty:
            # Prepare the dataset
            prepared_df, text_col, time_col = prepare_dataset_for_analysis(df)
            
            # Sample large datasets
            if len(prepared_df) > MAX_VIZ_SAMPLES:
                prepared_df = sample_large_dataset(prepared_df, MAX_VIZ_SAMPLES)
                st.warning(f"Dataset was sampled to {MAX_VIZ_SAMPLES} rows for visualization performance.")
            
            # Store in session state
            st.session_state.data = prepared_df
            st.session_state.text_column = text_col
            st.session_state.time_column = time_col
            st.session_state.last_update_time = datetime.now()
            
            return True
        else:
            st.error("Unable to load the dataset. Please check the file format.")
            return False

def create_sample_dataset(file_path):
    """Create a sample Twitter dataset for demo purposes."""
    # Create sample data
    np.random.seed(42)
    
    # Sample tweets
    sample_tweets = [
        "I absolutely love this new product! It's amazing! #happy",
        "Just had the worst customer service experience ever. Very disappointed. #angry",
        "This is okay, not great but not terrible either. #neutral",
        "Can't believe how awesome the new update is! Very impressed!",
        "Why is this service so slow? Frustrating to deal with.",
        "The weather is nice today. Going for a walk later.",
        "Super excited about the upcoming release! Can't wait! #excited",
        "This movie was terrible. Complete waste of money. #disappointed",
        "Just an average day. Nothing special to report.",
        "Best experience ever! Highly recommend to everyone! #love",
    ]
    
    # Generate more data by combining and varying the samples
    adjectives = ["really", "somewhat", "kind of", "absolutely", "barely", "hardly", "totally"]
    all_tweets = []
    
    for _ in range(500):
        base_tweet = np.random.choice(sample_tweets)
        
        # Sometimes modify the tweet
        if np.random.random() < 0.7:
            adj = np.random.choice(adjectives)
            words = base_tweet.split()
            insert_pos = min(np.random.randint(1, 4), len(words) - 1)
            words.insert(insert_pos, adj)
            modified_tweet = " ".join(words)
            all_tweets.append(modified_tweet)
        else:
            all_tweets.append(base_tweet)
    
    # Create timestamps (past 30 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    timestamps = [
        start_date + timedelta(
            seconds=np.random.randint(0, int((end_date - start_date).total_seconds()))
        )
        for _ in range(len(all_tweets))
    ]
    
    # Create DataFrame
    sample_df = pd.DataFrame({
        "tweet_text": all_tweets,
        "timestamp": timestamps,
        "user_id": np.random.randint(1000, 9999, size=len(all_tweets)),
        "platform": np.random.choice(["Twitter", "Facebook", "Instagram"], size=len(all_tweets)),
        "likes": np.random.randint(0, 1000, size=len(all_tweets)),
        "retweets": np.random.randint(0, 100, size=len(all_tweets))
    })
    
    # Save to CSV
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    sample_df.to_csv(file_path, index=False)
    
    return sample_df

def simulate_realtime_updates(data, chart_placeholder):
    """Simulate real-time data updates and visualization."""
    # Get the time column
    time_col = st.session_state.time_column
    
    # Create a copy of the data
    current_data = data.copy()
    
    # Sort by time
    if time_col in current_data.columns:
        current_data = current_data.sort_values(by=time_col)
    
    # Generate sentiment counts over time
    sentiment_counts = current_data.groupby([pd.Grouper(key=time_col, freq='1D'), 'sentiment']).size().unstack().fillna(0)
    
    # Create a real-time chart
    fig = px.line(
        sentiment_counts, 
        x=sentiment_counts.index, 
        y=sentiment_counts.columns,
        title="Sentiment Trends Over Time",
        labels={"value": "Count", "variable": "Sentiment"},
        color_discrete_map={
            "positive": "#2ECC71",
            "negative": "#E74C3C",
            "neutral": "#3498DB"
        }
    )
    
    # Simulate new data points
    last_time = current_data[time_col].max()
    new_time = last_time + timedelta(hours=np.random.randint(1, 6))
    
    # Random sentiment for new data
    sentiments = ["positive", "negative", "neutral"]
    weights = [0.4, 0.3, 0.3]  # Slightly biased towards positive
    
    # Add to the chart (this is just for visualization, not actually adding to the dataset)
    for _ in range(np.random.randint(3, 10)):
        new_sentiment = np.random.choice(sentiments, p=weights)
        
        # Find the latest value for this sentiment
        latest_val = 0
        if new_sentiment in sentiment_counts.columns and not sentiment_counts[new_sentiment].empty:
            latest_val = sentiment_counts[new_sentiment].iloc[-1]
            
        # Generate a new value with some randomness
        new_val = max(0, latest_val + np.random.randint(-5, 10))
        
        # Add annotation for new data point
        fig.add_annotation(
            x=new_time,
            y=new_val,
            text="New Data",
            showarrow=True,
            arrowhead=3,
            ax=0,
            ay=-40
        )
    
    # Update the chart
    chart_placeholder.plotly_chart(fig, use_container_width=True)

def analyze_sentiment(text):
    """Analyze sentiment of text using TextBlob"""
    blob = TextBlob(str(text))
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        return 'Positive'
    elif sentiment < 0:
        return 'Negative'
    return 'Neutral'

def generate_wordcloud(text_data):
    """Generate WordCloud from text data"""
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)
    return wordcloud

if __name__ == "__main__":
    main()