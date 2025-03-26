"""Utility functions for data loading and model operations."""

import os
import pandas as pd
import numpy as np
import json
from datetime import datetime
import time
import re
from pathlib import Path

def load_dataset(file_path):
    """
    Load dataset from file based on extension.
    
    Args:
        file_path (str): Path to the data file
        
    Returns:
        pandas.DataFrame: Loaded data
    """
    try:
        # Get file extension
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.csv':
            df = pd.read_csv(file_path)
        elif ext == '.xlsx':
            df = pd.read_excel(file_path)
        elif ext == '.json':
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return pd.DataFrame()

def detect_text_column(df):
    """Detect the column containing text data"""
    # Look for common text column names
    text_keywords = ['text', 'tweet', 'post', 'content', 'message', 'description']
    
    for col in df.columns:
        col_lower = col.lower()
        # Check if column name contains text keywords
        if any(keyword in col_lower for keyword in text_keywords):
            return col
        
        # Check if column contains string data
        if df[col].dtype == 'object':
            # Check if most values are strings with reasonable length
            sample = df[col].dropna().head(100)
            if sample.apply(lambda x: isinstance(x, str) and len(x) > 10).mean() > 0.8:
                return col
    
    return df.columns[0]  # Fallback to first column

def detect_datetime_column(df):
    """Detect the column containing datetime data"""
    # Look for common datetime column names
    time_keywords = ['time', 'date', 'timestamp', 'created', 'posted']
    
    for col in df.columns:
        col_lower = col.lower()
        # Check if column name contains time keywords
        if any(keyword in col_lower for keyword in time_keywords):
            try:
                pd.to_datetime(df[col])
                return col
            except:
                continue
        
        # Try to convert column to datetime
        if df[col].dtype == 'object':
            try:
                pd.to_datetime(df[col])
                return col
            except:
                continue
    
    return None

def prepare_dataset_for_analysis(df):
    """Prepare dataset for analysis"""
    # Detect text and time columns
    text_column = detect_text_column(df)
    time_column = detect_datetime_column(df)
    
    # Convert time column to datetime if it exists
    if time_column:
        try:
            df[time_column] = pd.to_datetime(df[time_column])
        except:
            time_column = None
    
    # Remove rows with missing text
    df = df.dropna(subset=[text_column])
    
    return df, text_column, time_column

def get_file_info(file_path):
    """
    Get information about a file.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        dict: File information
    """
    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    file_type = os.path.splitext(file_name)[1]
    
    return {
        "name": file_name,
        "size": file_size,
        "type": file_type,
        "created": datetime.fromtimestamp(os.path.getctime(file_path))
    }

def sample_large_dataset(df, max_samples):
    """Sample large dataset to manageable size"""
    if len(df) > max_samples:
        return df.sample(n=max_samples, random_state=42)
    return df