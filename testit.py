import streamlit
import transformers
import torch
import PIL
try:
    from summarizer import Summarizer
except ImportError:
    print("bert-extractive-summarizer not installed")
from datasets import load_dataset
from rouge_score import rouge_scorer
import pandas
import plotly
print("All dependencies imported successfully!")