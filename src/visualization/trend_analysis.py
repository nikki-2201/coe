import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy import stats

def detect_sentiment_trends(data, time_column, sentiment_column):
    """Detect trends in sentiment over time"""
    # Group by time and calculate sentiment metrics
    daily_sentiment = data.groupby(pd.Grouper(key=time_column, freq='1D')).agg({
        sentiment_column: ['count', 'value_counts']
    }).reset_index()
    
    # Calculate moving averages
    window_sizes = [3, 7, 14]  # Days
    trends = {}
    
    for window in window_sizes:
        ma = daily_sentiment[(sentiment_column, 'count')].rolling(window=window).mean()
        trends[f'{window}d_ma'] = ma
    
    # Detect significant changes
    changes = []
    baseline = daily_sentiment[(sentiment_column, 'count')].mean()
    threshold = baseline * 0.2  # 20% change threshold
    
    for i in range(1, len(daily_sentiment)):
        current = daily_sentiment[(sentiment_column, 'count')].iloc[i]
        previous = daily_sentiment[(sentiment_column, 'count')].iloc[i-1]
        
        if abs(current - previous) > threshold:
            changes.append({
                'date': daily_sentiment[time_column].iloc[i],
                'change': current - previous,
                'percentage': ((current - previous) / previous) * 100
            })
    
    return {
        'trends': trends,
        'changes': changes,
        'baseline': baseline
    }

def create_trend_chart(data, time_column, sentiment_column, trend_data):
    """Create chart showing sentiment trends and significant changes"""
    fig = go.Figure()
    
    # Add raw data
    fig.add_trace(
        go.Scatter(
            x=data[time_column],
            y=data[sentiment_column],
            name='Raw Sentiment',
            mode='markers',
            marker=dict(size=5, opacity=0.5)
        )
    )
    
    # Add moving averages
    colors = ['red', 'blue', 'green']
    for (window, ma), color in zip(trend_data['trends'].items(), colors):
        fig.add_trace(
            go.Scatter(
                x=data[time_column],
                y=ma,
                name=f'{window} Moving Average',
                line=dict(color=color)
            )
        )
    
    # Add significant changes
    if trend_data['changes']:
        change_dates = [change['date'] for change in trend_data['changes']]
        change_values = [change['percentage'] for change in trend_data['changes']]
        
        fig.add_trace(
            go.Scatter(
                x=change_dates,
                y=[trend_data['baseline']] * len(change_dates),
                mode='markers',
                marker=dict(
                    size=10,
                    symbol='star',
                    color='yellow',
                    line=dict(color='red', width=2)
                ),
                name='Significant Changes'
            )
        )
    
    fig.update_layout(
        title="Sentiment Trends Analysis",
        xaxis_title="Date",
        yaxis_title="Sentiment Score",
        height=500,
        showlegend=True
    )
    
    return fig

def calculate_trend_statistics(data, sentiment_column):
    """Calculate statistical measures of sentiment trends"""
    stats_dict = {
        'mean': np.mean(data[sentiment_column]),
        'median': np.median(data[sentiment_column]),
        'std': np.std(data[sentiment_column]),
        'skew': stats.skew(data[sentiment_column]),
        'kurtosis': stats.kurtosis(data[sentiment_column])
    }
    
    return stats_dict 