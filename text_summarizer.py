# venv/Scripts/text_summarizer.py
import streamlit as st
from transformers import BartForConditionalGeneration, BartTokenizer
from summarizer import Summarizer
from datasets import load_dataset
from rouge_score import rouge_scorer
import pandas as pd
import plotly.express as px

# Load models
@st.cache_resource
def load_models():
    bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    bert_model = Summarizer()
    return bart_model, bart_tokenizer, bert_model

# Abstractive summarization (BART)
def abstractive_summary(text, model, tokenizer, max_length=150, min_length=50):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=min_length,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Extractive summarization (BERT)
def extractive_summary(text, model, ratio=0.3):
    return model(text, ratio=ratio)

# Evaluate summaries
def evaluate_summaries(reference, extractive, abstractive):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    ext_scores = scorer.score(reference, extractive)
    abs_scores = scorer.score(reference, abstractive)
    return ext_scores, abs_scores

# Streamlit app
def main():
    st.title("Text Summarizer: Extractive vs. Abstractive")
    
    # Load models
    bart_model, bart_tokenizer, bert_model = load_models()
    
    # Input text
    input_text = st.text_area("Enter text to summarize:", height=200)
    use_sample = st.checkbox("Use sample CNN/DailyMail article")
    
    if st.button("Summarize"):
        if use_sample:
            try:
                dataset = load_dataset("cnn_dailymail", split="test")  # Omit version "3.0.0" to use latest
                input_text = dataset[0]["article"][:1000]
                reference_summary = dataset[0]["highlights"]
                st.write("Sample Article (truncated):")
                st.write(input_text[:500] + "...")
            except Exception as e:
                st.error(f"Error loading dataset: {e}")
                input_text = "Sample loading failed, please use custom text."
                reference_summary = None
        else:
            reference_summary = None
        
        # Generate summaries
        st.subheader("Extractive Summary (BERT)")
        ext_summary = extractive_summary(input_text, bert_model)
        st.write(ext_summary)
        
        st.subheader("Abstractive Summary (BART)")
        abs_summary = abstractive_summary(input_text, bart_model, bart_tokenizer)
        st.write(abs_summary)
        
        # Evaluate if using sample
        if reference_summary:
            ext_scores, abs_scores = evaluate_summaries(reference_summary, ext_summary, abs_summary)
            scores_df = pd.DataFrame({
                "Metric": ["ROUGE-1", "ROUGE-2", "ROUGE-L"],
                "Extractive": [ext_scores["rouge1"].fmeasure, ext_scores["rouge2"].fmeasure, ext_scores["rougeL"].fmeasure],
                "Abstractive": [abs_scores["rouge1"].fmeasure, abs_scores["rouge2"].fmeasure, abs_scores["rougeL"].fmeasure]
            })
            st.subheader("ROUGE Scores")
            fig = px.bar(scores_df, x="Metric", y=["Extractive", "Abstractive"], barmode="group",
                         title="ROUGE Scores: Extractive vs. Abstractive")
            st.plotly_chart(fig)

if __name__ == "__main__":
    main()