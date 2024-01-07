# SSIM916
Sentiment analysis of 50 articles from the Guardian regarding the COVID-19 pandemic inquiry.


# Guardian Articles Sentiment Analysis Model

## Introduction
This repository contains the code and model for sentiment analysis on Guardian articles related to the COVID-19 pandemic inquiry.

## Model
The final sentiment analysis model is based on Latent Dirichlet Allocation (LDA) for topic modeling and VADER for sentiment analysis.

### How to Use the Pickled Model
1. Install the required libraries: `pip install pandas nltk gensim wordcloud matplotlib scikit-learn beautifulsoup4`
2. Load the pickled model in your Python script or notebook:
    ```python
    import pickle

    with open('final_model.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)
    ```
3. Use the loaded model to analyze new data.

## Code
The `guardian_sentiment_analysis.ipynb` notebook contains the complete code for data collection, preprocessing, analysis, and model training.

## Data
The `guardian_articles_data.csv` file contains the processed data used for model training.

Feel free to contact the author for any questions or clarifications.
