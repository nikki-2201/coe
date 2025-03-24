# Social Media Sentiment Analysis Platform

A real-time social media post tracking and sentiment analysis platform built with Python and Streamlit. This application allows users to upload datasets containing social media posts and perform sentiment analysis on them.

## Features

- Upload CSV files containing social media posts
- Perform sentiment analysis on text data
- Interactive visualizations:
  - Sentiment distribution pie chart
  - Word cloud visualization
- Download analysis results
- User-friendly interface

## Installation

1. Clone this repository:

```bash
git clone <repository-url>
cd social-media-sentiment-analysis
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:

```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

3. Upload your CSV file containing social media posts. The file should have at least one column containing the text data to analyze.

4. Select the column containing the text data from the dropdown menu.

5. Click "Analyze Sentiment" to start the analysis.

6. View the results and download them if needed.

## Input Data Format

The application accepts CSV files with the following requirements:

- At least one column containing text data (social media posts)
- The CSV file should be properly formatted
- Text data should be in a readable format

## Technologies Used

- Python 3.8+
- Streamlit
- TextBlob for sentiment analysis
- NLTK for text processing
- Plotly for interactive visualizations
- WordCloud for word cloud generation
- Pandas for data manipulation

## License

This project is licensed under the MIT License - see the LICENSE file for details.
