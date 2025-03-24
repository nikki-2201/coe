import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import numpy as np

def create_sentiment_distribution_chart(sentiment_data):
    """Create pie chart of sentiment distribution"""
    sentiment_counts = sentiment_data.value_counts()
    
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="Sentiment Distribution",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_layout(
        showlegend=True,
        legend_title="Sentiment",
        height=400
    )
    
    return fig

def create_sentiment_over_time_chart(data, time_column, sentiment_column):
    """Create line chart of sentiment trends over time"""
    # Group by time and sentiment
    sentiment_over_time = data.groupby(
        [pd.Grouper(key=time_column, freq='1D'), sentiment_column]
    ).size().unstack().fillna(0)
    
    # Create line chart
    fig = go.Figure()
    
    for sentiment in sentiment_over_time.columns:
        fig.add_trace(
            go.Scatter(
                x=sentiment_over_time.index,
                y=sentiment_over_time[sentiment],
                name=sentiment,
                mode='lines+markers'
            )
        )
    
    fig.update_layout(
        title="Sentiment Trends Over Time",
        xaxis_title="Date",
        yaxis_title="Count",
        height=400,
        showlegend=True,
        legend_title="Sentiment"
    )
    
    return fig

def create_sentiment_wordcloud(text_data, sentiment_data=None, sentiment_value=None):
    """Create word cloud visualization"""
    if sentiment_data is not None and sentiment_value is not None:
        # Filter text for specific sentiment
        text_data = text_data[sentiment_data == sentiment_value]
    
    # Combine all text
    text = " ".join(text_data.astype(str))
    
    # Create and generate word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_words=100,
        colormap='viridis'
    ).generate(text)
    
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    
    return fig

def create_sentiment_heatmap(data, time_column):
    """Create heatmap of sentiment by hour and day"""
    # Extract hour and day of week
    data['hour'] = data[time_column].dt.hour
    data['day_of_week'] = data[time_column].dt.day_name()
    
    # Create pivot table
    heatmap_data = pd.pivot_table(
        data,
        values='sentiment_score',  # Assuming numerical sentiment scores
        index='day_of_week',
        columns='hour',
        aggfunc='mean'
    )
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='RdYlBu'
    ))
    
    fig.update_layout(
        title="Sentiment by Hour and Day",
        xaxis_title="Hour of Day",
        yaxis_title="Day of Week",
        height=400
    )
    
    return fig

def create_top_terms_chart(text_data, sentiment_data, n_terms=10):
    """Create bar chart of top terms by sentiment"""
    # Split text into words and count frequencies
    word_counts = Counter(" ".join(text_data).lower().split())
    
    # Create DataFrame of word frequencies
    term_freq = pd.DataFrame.from_dict(word_counts, orient='index', columns=['count'])
    term_freq = term_freq.sort_values('count', ascending=False)
    
    # Create bar chart
    fig = px.bar(
        term_freq.head(n_terms),
        title=f"Top {n_terms} Terms",
        labels={'index': 'Term', 'count': 'Frequency'}
    )
    
    fig.update_layout(
        showlegend=False,
        height=400,
        xaxis_tickangle=-45
    )
    
    return fig 
 
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import numpy as np

def create_sentiment_distribution_chart(sentiment_data):
    """Create pie chart of sentiment distribution"""
    sentiment_counts = sentiment_data.value_counts()
    
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="Sentiment Distribution",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_layout(
        showlegend=True,
        legend_title="Sentiment",
        height=400
    )
    
    return fig

def create_sentiment_over_time_chart(data, time_column, sentiment_column):
    """Create line chart of sentiment trends over time"""
    # Group by time and sentiment
    sentiment_over_time = data.groupby(
        [pd.Grouper(key=time_column, freq='1D'), sentiment_column]
    ).size().unstack().fillna(0)
    
    # Create line chart
    fig = go.Figure()
    
    for sentiment in sentiment_over_time.columns:
        fig.add_trace(
            go.Scatter(
                x=sentiment_over_time.index,
                y=sentiment_over_time[sentiment],
                name=sentiment,
                mode='lines+markers'
            )
        )
    
    fig.update_layout(
        title="Sentiment Trends Over Time",
        xaxis_title="Date",
        yaxis_title="Count",
        height=400,
        showlegend=True,
        legend_title="Sentiment"
    )
    
    return fig

def create_sentiment_wordcloud(text_data, sentiment_data=None, sentiment_value=None):
    """Create word cloud visualization"""
    if sentiment_data is not None and sentiment_value is not None:
        # Filter text for specific sentiment
        text_data = text_data[sentiment_data == sentiment_value]
    
    # Combine all text
    text = " ".join(text_data.astype(str))
    
    # Create and generate word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_words=100,
        colormap='viridis'
    ).generate(text)
    
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    
    return fig

def create_sentiment_heatmap(data, time_column):
    """Create heatmap of sentiment by hour and day"""
    # Extract hour and day of week
    data['hour'] = data[time_column].dt.hour
    data['day_of_week'] = data[time_column].dt.day_name()
    
    # Create pivot table
    heatmap_data = pd.pivot_table(
        data,
        values='sentiment_score',  # Assuming numerical sentiment scores
        index='day_of_week',
        columns='hour',
        aggfunc='mean'
    )
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='RdYlBu'
    ))
    
    fig.update_layout(
        title="Sentiment by Hour and Day",
        xaxis_title="Hour of Day",
        yaxis_title="Day of Week",
        height=400
    )
    
    return fig

def create_top_terms_chart(text_data, sentiment_data, n_terms=10):
    """Create bar chart of top terms by sentiment"""
    # Split text into words and count frequencies
    word_counts = Counter(" ".join(text_data).lower().split())
    
    # Create DataFrame of word frequencies
    term_freq = pd.DataFrame.from_dict(word_counts, orient='index', columns=['count'])
    term_freq = term_freq.sort_values('count', ascending=False)
    
    # Create bar chart
    fig = px.bar(
        term_freq.head(n_terms),
        title=f"Top {n_terms} Terms",
        labels={'index': 'Term', 'count': 'Frequency'}
    )
    
    fig.update_layout(
        showlegend=False,
        height=400,
        xaxis_tickangle=-45
    )
    
    return fig