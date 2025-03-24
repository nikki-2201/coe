import streamlit as st
import pandas as pd
from .visualization.sentiment_plots import (
    create_sentiment_distribution_chart,
    create_sentiment_wordcloud,
    create_sentiment_over_time_chart,
    create_sentiment_heatmap,
    create_top_terms_chart
)
from .visualization.trend_analysis import (
    detect_sentiment_trends,
    create_trend_chart,
    calculate_trend_statistics
)

def show_dataset_info(data, text_column, time_column):
    """Display dataset information"""
    st.subheader("📊 Dataset Information")
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Basic Statistics:")
        st.write(f"- Total Records: {len(data):,}")
        st.write(f"- Text Column: {text_column}")
        if time_column:
            st.write(f"- Time Range: {data[time_column].min()} to {data[time_column].max()}")
    
    with col2:
        st.write("Sample Data:")
        st.dataframe(data.head(5))

def show_sentiment_analysis_section(data):
    """Display sentiment analysis results"""
    st.subheader("🎯 Sentiment Analysis Results")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Distribution", "Word Cloud", "Top Terms"])
    
    with tab1:
        # Sentiment distribution
        st.plotly_chart(
            create_sentiment_distribution_chart(data['sentiment']),
            use_container_width=True
        )
    
    with tab2:
        # Word clouds for each sentiment
        sentiments = data['sentiment'].unique()
        selected_sentiment = st.selectbox(
            "Select sentiment to view word cloud:",
            sentiments
        )
        
        fig = create_sentiment_wordcloud(
            data['processed_text'],
            data['sentiment'],
            selected_sentiment
        )
        st.pyplot(fig)
    
    with tab3:
        # Top terms
        n_terms = st.slider("Number of terms to show:", 5, 20, 10)
        st.plotly_chart(
            create_top_terms_chart(
                data['processed_text'],
                data['sentiment'],
                n_terms
            ),
            use_container_width=True
        )

def show_trend_analysis_section(data, time_column):
    """Display trend analysis results"""
    st.subheader("📈 Trend Analysis")
    
    # Detect trends
    trend_data = detect_sentiment_trends(
        data,
        time_column,
        'sentiment_score'  # Assuming numerical sentiment scores
    )
    
    # Create trend chart
    st.plotly_chart(
        create_trend_chart(
            data,
            time_column,
            'sentiment_score',
            trend_data
        ),
        use_container_width=True
    )
    
    # Show statistics
    stats = calculate_trend_statistics(data, 'sentiment_score')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Statistical Measures:")
        st.write(f"- Mean: {stats['mean']:.3f}")
        st.write(f"- Median: {stats['median']:.3f}")
        st.write(f"- Standard Deviation: {stats['std']:.3f}")
    
    with col2:
        st.write("Distribution Measures:")
        st.write(f"- Skewness: {stats['skew']:.3f}")
        st.write(f"- Kurtosis: {stats['kurtosis']:.3f}")
    
    # Show significant changes
    if trend_data['changes']:
        st.subheader("⚠️ Significant Changes Detected")
        for change in trend_data['changes']:
            st.write(
                f"- {change['date'].strftime('%Y-%m-%d')}: "
                f"{change['percentage']:.1f}% change"
            )
    
    # Show hourly patterns
    st.subheader("⏰ Hourly Patterns")
    st.plotly_chart(
        create_sentiment_heatmap(data, time_column),
        use_container_width=True
    ) 