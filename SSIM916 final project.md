```python
## Importing libraries

import json
import requests
import pandas as pd
import os
from bs4 import BeautifulSoup
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora
from gensim.models import LdaModel
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
```


```python
## Fetching data

# API key
api_key = "9e7f1b62-3094-4d2a-80a1-a0adde906678"

# Fetch 50 articles
query = "COVID-19 pandemic inquiry"
url = f'https://content.guardianapis.com/search?q={query}&api-key={api_key}&page-size=50'
response = requests.get(url)
data = response.json()
articles = data.get("response", {}).get("results", [])
```


```python
## Text preprocessing and sentiment analysis setup

# VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')

# Define function for text preprocessing
def preprocess_text(text, additional_stopwords=[]):
    soup = BeautifulSoup(text, 'html.parser')
    clean_text = re.sub(r'[^a-zA-Z\s]', '', soup.get_text(separator=' '))
    clean_text = re.sub(r'\s+', ' ', clean_text).strip() # Remove white space around text

    tokens = word_tokenize(clean_text)
    tokens = [token.lower() for token in tokens if token.isalpha()]

    stop_words = set(stopwords.words('english'))

    # Remove proper nouns from the list of stopwords
    stop_words = stop_words.union(set(additional_stopwords))

    # Keep only adjectives and nouns
    pos_tags = nltk.pos_tag(tokens)
    tokens = [token for token, pos in pos_tags if pos.startswith('J') or pos.startswith('N')]

    tokens = [token for token in tokens if token not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return ' '.join(tokens)

# Create a list to store processed article data
processed_article_data_list = []

# Create a sentiment intensity analyzer
sia = SentimentIntensityAnalyzer()

# Additional stopwords to exclude proper nouns and some common words 
additional_stopwords = ['pandemic', 'covid', 'uk', 'edition', 'news', 'newsletter', 'job', 'guardian', 'yousaf', 'boris', 'johnson', 'inquiry', 'matt', 'hancock', 'dominic', 'cummings', 'rishi', 'sunak', 'chris', 'keith']

# Iterate through articles
for article in articles:
    pub_date = article["webPublicationDate"]
    summary = article["webTitle"]

    # Fetch the content from the article's URL
    article_url = article["webUrl"]
    article_response = requests.get(article_url)

    # Check the content type
    content_type = article_response.headers.get('content-type', '').lower()

    try:
        if 'application/json' in content_type:
            article_data = article_response.json()
            article_content = article_data.get("response", {}).get("content", {}).get("blocks", {}).get("body", "")
        else:
            article_content = article_response.text

        if article_content:
            # Perform text preprocessing with additional stopwords
            preprocessed_text = preprocess_text(article_content, additional_stopwords)

            # Perform sentiment analysis
            sentiment_score = sia.polarity_scores(preprocessed_text)

            # Convert publication date to datetime format
            pub_date = pd.to_datetime(pub_date)

            # Store data in a dictionary
            processed_article_data_dict = {
                'Publication Date': pub_date,
                'Summary': summary,
                'Processed Text': preprocessed_text,
                'Sentiment Score': sentiment_score['compound']
            }

            # Append the dictionary to the list
            processed_article_data_list.append(processed_article_data_dict)

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")

# Create a data frame from the list of processed article data and print
df_processed_text = pd.DataFrame(processed_article_data_list)
print("Processed Article Data DataFrame:")
print(df_processed_text)
```

    [nltk_data] Downloading package vader_lexicon to
    [nltk_data]     C:\Users\lundt\AppData\Roaming\nltk_data...
    [nltk_data]   Package vader_lexicon is already up-to-date!
    

    Processed Article Data DataFrame:
                Publication Date  \
    0  2023-12-06 06:54:24+00:00   
    1  2023-10-04 17:11:55+00:00   
    2  2023-12-08 16:22:02+00:00   
    3  2023-11-02 18:39:22+00:00   
    4  2023-11-14 17:33:06+00:00   
    5  2023-09-21 01:13:11+00:00   
    6  2023-11-21 11:00:13+00:00   
    7  2023-09-20 20:50:39+00:00   
    8  2023-09-17 06:00:53+00:00   
    9  2023-12-07 17:07:49+00:00   
    10 2023-12-03 08:00:09+00:00   
    11 2023-11-03 14:41:30+00:00   
    12 2023-09-21 07:06:21+00:00   
    13 2023-12-06 20:27:34+00:00   
    14 2023-06-14 14:53:25+00:00   
    15 2023-10-31 19:08:46+00:00   
    16 2023-06-13 18:08:16+00:00   
    17 2023-12-11 18:21:39+00:00   
    18 2023-12-02 13:29:49+00:00   
    19 2023-10-06 08:51:48+00:00   
    20 2023-11-20 23:17:46+00:00   
    21 2023-12-02 07:27:46+00:00   
    22 2023-12-11 16:58:01+00:00   
    23 2023-12-07 17:29:17+00:00   
    24 2023-12-06 07:00:04+00:00   
    25 2023-12-15 16:00:32+00:00   
    26 2023-10-31 09:24:58+00:00   
    27 2023-11-01 14:03:38+00:00   
    28 2023-10-18 05:45:32+00:00   
    29 2023-11-20 17:23:50+00:00   
    30 2023-11-27 16:27:57+00:00   
    31 2023-12-01 16:56:20+00:00   
    32 2023-11-30 17:30:15+00:00   
    33 2023-10-30 05:00:04+00:00   
    34 2023-11-21 16:01:06+00:00   
    35 2023-12-20 15:40:03+00:00   
    36 2023-11-14 16:03:57+00:00   
    37 2023-11-02 19:29:17+00:00   
    38 2023-02-16 18:57:47+00:00   
    39 2023-12-09 09:00:16+00:00   
    40 2023-11-22 11:11:31+00:00   
    41 2023-06-13 05:43:33+00:00   
    42 2023-11-26 07:00:13+00:00   
    43 2023-12-04 11:10:30+00:00   
    44 2023-11-03 08:00:04+00:00   
    45 2023-11-01 20:06:07+00:00   
    46 2023-06-01 18:00:37+00:00   
    47 2023-11-29 18:35:33+00:00   
    48 2023-10-17 16:09:45+00:00   
    49 2023-11-09 17:55:53+00:00   
    
                                                  Summary  \
    0   Boris Johnson faces tough questions at Covid i...   
    1   Children were failed by pandemic policies, Cov...   
    2    ‘Completely out of touch’: five people hit ha...   
    3   The Guardian view on the Covid-19 inquiry: a w...   
    4   Local expertise was ignored during Covid pande...   
    5   Covid-19 inquiry will exclude state and territ...   
    6   What does the Covid inquiry really tell us? If...   
    7   Morning Mail: AFL’s gambling problem, Covid-19...   
    8   Sunak and Johnson Covid-19 inquiry appearances...   
    9   Boris Johnson tells Covid inquiry he avoided e...   
    10   Covid inquiry: 10 questions facing Boris Johnson   
    11  Another truth from the Covid inquiry: women we...   
    12  Peak medical body labels Covid-19 review ‘half...   
    13     Boris Johnson at the Covid inquiry: key points   
    14  Covid testing was a weakness in early pandemic...   
    15  The Guardian view on the Covid inquiry: shocki...   
    16  ‘A bowl of spaghetti’ : Covid inquiry opens wi...   
    17  ‘Deeply sorry’: what Rishi Sunak said to the C...   
    18  Covid inquiry: Boris Johnson ‘to admit he made...   
    19  Wetherspoon’s returns to profit for first time...   
    20  Patrick Vallance contradicts Rishi Sunak’s evi...   
    21  Boris Johnson’s legacy will be shaped by Covid...   
    22  Boris Johnson’s negligence has been laid bare ...   
    23  Boris Johnson’s second day at the Covid inquir...   
    24  Boris Johnson ‘trying to rewrite history’ befo...   
    25  Hugo Keith KC: dogged fact-finder at heart of ...   
    26  Brexit prioritised over tackling Covid at star...   
    27  ‘Absence of humanity’: Helen MacNamara’s evide...   
    28  Wednesday briefing: What you’ve missed at the ...   
    29  What we learned from Patrick Vallance at the C...   
    30  UK Covid response was London-centric, Andy Bur...   
    31  Boris Johnson wanted ‘massive fines’ for lockd...   
    32  Matt Hancock ‘was not told about eat out to he...   
    33  UK Covid inquiry reaches week of potentially d...   
    34  I regret saying ‘herd immunity’, Vallance tell...   
    35  Women’s voices barely heard in Boris Johnson’s...   
    36  Incarcerated students earn degrees in groundbr...   
    37  What we learned about Matt Hancock from the Co...   
    38  PR firm behind Tory pandemic response linked t...   
    39  Eating out and Partygate: Covid inquiry questi...   
    40  Herd immunity was never UK policy, Chris Whitt...   
    41  Tuesday briefing: A years-long inquiry into th...   
    42  Stretched NHS even less ready to cope with a n...   
    43  Key questions Boris Johnson is likely to face ...   
    44  Macho posturing costs lives: another lesson fr...   
    45  Dead cats and hairdryers: Dominic Cummings’ ev...   
    46  The Guardian view on the Covid-19 standoff: it...   
    47  Rishi Sunak pushed hard for lifting of Covid r...   
    48  Communication with ministers was poor, scienti...   
    49  Yousaf apologises for ‘shortcomings’ with rele...   
    
                                           Processed Text  Sentiment Score  
    0   tough question handling skip main content skip...          -0.8402  
    1   child policy skip main content skip navigation...           0.9371  
    2   touch people skip main content skip navigation...          -0.9349  
    3   view week deep editorial skip main content ski...           0.8360  
    4   local expertise skip main content skip navigat...          -0.9064  
    5   state territory decision albanese australia sk...           0.9786  
    6   tomorrow nothing better zoe skip main content ...          -0.9765  
    7   morning mail afl problem uk green retreat aust...          -0.9962  
    8   appearance party conference season skip main c...          -0.5719  
    9   devolved administration political reason skip ...          -0.9671  
    10  question skip main content skip navigation ski...          -0.9382  
    11  truth woman ppe caroline criado skip main cont...           0.5423  
    12  peak medical body label review albanese govern...          -0.4145  
    13  key point skip main content skip navigation sk...          -0.9966  
    14  testing weakness early response dhsc tell skip...          -0.0772  
    15  view failure spotlight editorial skip main con...          -0.9955  
    16  bowl spaghetti flowchart uk planning skip main...          -0.9783  
    17  sorry skip main content skip navigation skip p...           0.9265  
    18  mistake skip main content skip navigation skip...          -0.9688  
    19  wetherspoons return first time jd skip main co...           0.9829  
    20  patrick vallance contradicts sunaks evidence s...           0.9732  
    21  johnson legacy appearance skip main content sk...          -0.3818  
    22  johnson negligence bare skip main content skip...          -0.7579  
    23  johnson second day key point skip main content...           0.5020  
    24  history appearance skip main content skip navi...          -0.9360  
    25  hugo kc factfinder heart skip main content ski...          -0.1531  
    26  brexit start exminister coronavirus skip main ...          -0.9917  
    27  absence humanity macnamaras evidence skip main...           0.9718  
    28  wednesday youve skip main content skip navigat...          -0.9973  
    29  patrick vallance skip main content skip naviga...           0.9735  
    30  response londoncentric andy burnham tell skip ...           0.6369  
    31  massive fine lockdown breach hears skip main c...           0.3400  
    32  eat scheme day politics skip main content skip...          -0.8834  
    33  week testimony skip main content skip navigati...           0.6815  
    34  herd immunity vallance tell skip main content ...          -0.9864  
    35  woman voice johnson skip main content skip nav...          -0.9563  
    36  incarcerated student degree university program...          -0.9957  
    37  politics skip main content skip navigation ski...           0.7506  
    38  pr firm tory response skip main content skip n...           0.9349  
    39  question skip main content skip navigation ski...          -0.9580  
    40  herd immunity policy whitty skip main content ...          -0.9524  
    41  tuesday yearslong start today skip main conten...           0.6560  
    42  ready new scientist health skip main content s...          -0.9423  
    43  key question likely skip main content skip nav...          -0.9657  
    44  macho posturing cost lesson gaby skip main con...          -0.9611  
    45  dead cat hairdryers evidence skip main content...          -0.9939  
    46  view standoff matter editorial skip main conte...          -0.5859  
    47  rule hears skip main content skip navigation s...          -0.7717  
    48  communication minister poor scientist tell ski...          -0.8994  
    49  apologises shortcoming release whatsapp messag...          -0.2937  
    


```python
## TF-IDF vectorisation, analysis and visualitation, and data combination

# Combine all processed texts into a single string
all_processed_text = ' '.join(df_processed_text['Processed Text'])

# TF-IDF vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([all_processed_text])

# Get feature names (words)
feature_names = vectorizer.get_feature_names_out()

# Convert TF-IDF matrix to data frame
df_tfidf_data = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

# Merge the data frames by the common column (e.g., index)
df_combined_data = pd.merge(df_processed_text, df_tfidf_data, left_index=True, right_index=True)

# Sort by sentiment score in descending order and print
df_combined_data = df_combined_data.sort_values(by='Sentiment Score', ascending=False)
print("Refined Combined Data DataFrame:")
print(df_combined_data)
```

    Refined Combined Data DataFrame:
               Publication Date  \
    0 2023-12-06 06:54:24+00:00   
    
                                                 Summary  \
    0  Boris Johnson faces tough questions at Covid i...   
    
                                          Processed Text  Sentiment Score  \
    0  tough question handling skip main content skip...          -0.8402   
    
           abba   ability     able  abortion  aboutturn   absence  ...   youtube  \
    0  0.000744  0.002976  0.01265  0.002976   0.000744  0.005953  ...  0.040182   
    
          youve       yui    zayidi    zealot      zero       zoe       zoo  \
    0  0.002976  0.001488  0.000744  0.002232  0.000744  0.002232  0.000744   
    
           zoom  zoonotic  
    0  0.001488  0.000744  
    
    [1 rows x 3597 columns]
    


```python
## Tokenisation and LDA topic modelling - 20 keywords 

# Tokenize the processed text
df_processed_text['Tokenized Text'] = df_processed_text['Processed Text'].apply(word_tokenize)

# Create a dictionary representation of the documents
dictionary = corpora.Dictionary(df_processed_text['Tokenized Text'])

# Create a bag-of-words representation 
corpus = [dictionary.doc2bow(text) for text in df_processed_text['Tokenized Text']]

# Set the number of topics and build the LDA model
num_topics = 5
lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

# Get topics for each document
df_processed_text['Topics'] = df_processed_text['Tokenized Text'].apply(lambda x: lda_model.get_document_topics(dictionary.doc2bow(x)))

# Extract the top 5 keywords for each topic
topics_keywords = []
for topic_id in range(num_topics):
    keywords = lda_model.show_topic(topic_id, topn=10)
    topics_keywords.append([word for word, _ in keywords])

# Add topics and keywords to the data frame
df_processed_text['Topics and Keywords'] = df_processed_text['Topics'].apply(lambda x: [topics_keywords[topic_id] for topic_id, _ in x])

# Print with topics and keywords
print("DataFrame with Topics and Keywords:")
print(df_processed_text[['Summary', 'Topics and Keywords']])

# Merge topics and keywords into a single list of keywords
all_keywords = [keyword for sublist in topics_keywords for keyword in sublist]

# Identify top N words with highest TF-IDF scores
top_n = 20
top_words = df_tfidf_data.sum().nlargest(top_n)

# Visualisation of top words
plt.figure(figsize=(20, 6))
top_words.plot(kind='barh', color='skyblue')
plt.title(f'Top {top_n} Words with Highest TF-IDF Scores')
plt.xlabel('TF-IDF Score')
plt.ylabel('Words')
plt.show()
print("Top Words with Highest TF-IDF Scores:")
print(top_words)
```

    DataFrame with Topics and Keywords:
                                                  Summary  \
    0   Boris Johnson faces tough questions at Covid i...   
    1   Children were failed by pandemic policies, Cov...   
    2    ‘Completely out of touch’: five people hit ha...   
    3   The Guardian view on the Covid-19 inquiry: a w...   
    4   Local expertise was ignored during Covid pande...   
    5   Covid-19 inquiry will exclude state and territ...   
    6   What does the Covid inquiry really tell us? If...   
    7   Morning Mail: AFL’s gambling problem, Covid-19...   
    8   Sunak and Johnson Covid-19 inquiry appearances...   
    9   Boris Johnson tells Covid inquiry he avoided e...   
    10   Covid inquiry: 10 questions facing Boris Johnson   
    11  Another truth from the Covid inquiry: women we...   
    12  Peak medical body labels Covid-19 review ‘half...   
    13     Boris Johnson at the Covid inquiry: key points   
    14  Covid testing was a weakness in early pandemic...   
    15  The Guardian view on the Covid inquiry: shocki...   
    16  ‘A bowl of spaghetti’ : Covid inquiry opens wi...   
    17  ‘Deeply sorry’: what Rishi Sunak said to the C...   
    18  Covid inquiry: Boris Johnson ‘to admit he made...   
    19  Wetherspoon’s returns to profit for first time...   
    20  Patrick Vallance contradicts Rishi Sunak’s evi...   
    21  Boris Johnson’s legacy will be shaped by Covid...   
    22  Boris Johnson’s negligence has been laid bare ...   
    23  Boris Johnson’s second day at the Covid inquir...   
    24  Boris Johnson ‘trying to rewrite history’ befo...   
    25  Hugo Keith KC: dogged fact-finder at heart of ...   
    26  Brexit prioritised over tackling Covid at star...   
    27  ‘Absence of humanity’: Helen MacNamara’s evide...   
    28  Wednesday briefing: What you’ve missed at the ...   
    29  What we learned from Patrick Vallance at the C...   
    30  UK Covid response was London-centric, Andy Bur...   
    31  Boris Johnson wanted ‘massive fines’ for lockd...   
    32  Matt Hancock ‘was not told about eat out to he...   
    33  UK Covid inquiry reaches week of potentially d...   
    34  I regret saying ‘herd immunity’, Vallance tell...   
    35  Women’s voices barely heard in Boris Johnson’s...   
    36  Incarcerated students earn degrees in groundbr...   
    37  What we learned about Matt Hancock from the Co...   
    38  PR firm behind Tory pandemic response linked t...   
    39  Eating out and Partygate: Covid inquiry questi...   
    40  Herd immunity was never UK policy, Chris Whitt...   
    41  Tuesday briefing: A years-long inquiry into th...   
    42  Stretched NHS even less ready to cope with a n...   
    43  Key questions Boris Johnson is likely to face ...   
    44  Macho posturing costs lives: another lesson fr...   
    45  Dead cats and hairdryers: Dominic Cummings’ ev...   
    46  The Guardian view on the Covid-19 standoff: it...   
    47  Rishi Sunak pushed hard for lifting of Covid r...   
    48  Communication with ministers was poor, scienti...   
    49  Yousaf apologises for ‘shortcomings’ with rele...   
    
                                      Topics and Keywords  
    0   [[job, dec, opinion, search, culture, people, ...  
    1   [[job, opinion, government, search, nov, cultu...  
    2   [[job, dec, opinion, search, culture, people, ...  
    3   [[job, dec, opinion, search, culture, people, ...  
    4   [[job, opinion, government, search, nov, cultu...  
    5   [[state, australia, health, government, job, n...  
    6   [[job, dec, opinion, search, culture, people, ...  
    7   [[job, opinion, government, search, nov, cultu...  
    8   [[job, dec, opinion, search, culture, people, ...  
    9   [[job, dec, opinion, search, culture, people, ...  
    10  [[job, dec, opinion, search, culture, people, ...  
    11  [[opinion, job, view, search, health, woman, s...  
    12  [[state, australia, health, government, job, n...  
    13  [[job, dec, opinion, search, culture, people, ...  
    14  [[job, dec, opinion, search, culture, people, ...  
    15  [[job, dec, opinion, search, culture, people, ...  
    16  [[job, opinion, government, search, nov, cultu...  
    17  [[job, dec, opinion, search, culture, people, ...  
    18  [[job, dec, opinion, search, culture, people, ...  
    19  [[state, australia, health, government, job, n...  
    20  [[job, dec, opinion, search, culture, people, ...  
    21  [[job, dec, opinion, search, culture, people, ...  
    22  [[job, dec, opinion, search, culture, people, ...  
    23  [[job, dec, opinion, search, culture, people, ...  
    24  [[job, dec, opinion, search, culture, people, ...  
    25  [[job, dec, opinion, search, culture, people, ...  
    26  [[job, dec, opinion, search, culture, people, ...  
    27  [[state, australia, health, government, job, n...  
    28  [[job, opinion, government, search, nov, cultu...  
    29  [[job, dec, opinion, search, culture, people, ...  
    30  [[job, dec, opinion, search, culture, people, ...  
    31  [[job, dec, opinion, search, culture, people, ...  
    32  [[job, opinion, government, search, nov, cultu...  
    33  [[job, dec, opinion, search, culture, people, ...  
    34  [[job, dec, opinion, search, culture, people, ...  
    35  [[job, dec, opinion, search, culture, people, ...  
    36  [[job, opinion, government, search, nov, cultu...  
    37  [[job, dec, opinion, search, culture, people, ...  
    38  [[job, dec, opinion, search, culture, people, ...  
    39  [[job, dec, opinion, search, culture, people, ...  
    40  [[job, dec, opinion, search, culture, people, ...  
    41  [[opinion, job, view, search, health, woman, s...  
    42  [[job, opinion, government, search, nov, cultu...  
    43  [[job, dec, opinion, search, culture, people, ...  
    44  [[job, dec, opinion, search, culture, people, ...  
    45  [[job, dec, opinion, search, culture, people, ...  
    46  [[job, opinion, government, search, nov, cultu...  
    47  [[job, dec, opinion, search, culture, people, ...  
    48  [[job, dec, opinion, search, culture, people, ...  
    49  [[job, opinion, government, search, nov, cultu...  
    


    
![png](output_4_1.png)
    


    Top Words with Highest TF-IDF Scores:
    job           4.074274
    opinion       3.834328
    dec           3.688003
    search        3.371756
    culture       3.131940
    sport         2.777999
    government    2.772370
    health        2.572321
    view          2.554639
    policy        2.544336
    minister      2.496597
    newsletter    2.465669
    vallance      2.438139
    people        2.436951
    evidence      2.412702
    skip          2.205951
    digital       2.112558
    politics      2.081555
    print         2.030350
    archive       2.023054
    dtype: float64
    


```python
## Sentiment analysis visualisation

# Function to perform sentiment analysis on a given text
def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = analyzer.polarity_scores(text)['compound']
    return sentiment_score

# Function to generate and display a word cloud for a given sentiment range
def generate_wordcloud_for_sentiment(sentiment_range):
    sentiment_labels = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
    sentiment_label = sentiment_labels[sentiment_ranges.index(sentiment_range)]
    
    cluster_text = df_processed_text[(df_processed_text['Sentiment Score'] >= sentiment_range[0]) & 
                                      (df_processed_text['Sentiment Score'] < sentiment_range[1])]['Processed Text'].str.cat(sep=' ')
    
    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cluster_text)
    
    # Display the word cloud using matplotlib
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Word Cloud for {sentiment_label} Sentiment')
    plt.axis('off')
    
    # Display articles contributing to the sentiment range
    contributing_articles = df_processed_text[(df_processed_text['Sentiment Score'] >= sentiment_range[0]) & 
                                             (df_processed_text['Sentiment Score'] < sentiment_range[1])][['Publication Date', 'Summary']]
    print(f"Articles contributing to {sentiment_label} Sentiment:\n")
    print(contributing_articles)
    
    plt.show()

# Function to create a scatter plot of sentiment scores over time
def scatter_plot_sentiment_over_time():
    tokenized_docs = [text.split() for text in df_processed_text['Processed Text']]
    dictionary = Dictionary(tokenized_docs)

    df_processed_text.sort_values(by='Publication Date', inplace=True)

    sentiment_ranges = [(-1.0, -0.6), (-0.6, -0.2), (-0.2, 0.2), (0.2, 0.6), (0.6, 1.0)]

    plt.figure(figsize=(15, 8))
    line_dict = {}

    for sentiment_range, color in zip(sentiment_ranges, ['red', 'orange', 'yellow', 'green', 'blue']):
        filtered_articles = df_processed_text[(df_processed_text['Sentiment Score'] >= sentiment_range[0]) &
                                              (df_processed_text['Sentiment Score'] < sentiment_range[1])]

        filtered_articles.sort_values(by='Publication Date', inplace=True)
        x_line = []
        y_line = []

        for idx, article in filtered_articles.iterrows():
            current_corpus = [dictionary.doc2bow(article['Processed Text'].split())]
            lda_gensim = LdaModel(corpus=current_corpus, num_topics=1, id2word=dictionary, random_state=42)

            x_line.append(article['Publication Date'])
            y_line.append(article['Sentiment Score'])

            plt.scatter(article['Publication Date'], article['Sentiment Score'], color=color)

        line_dict[color] = (x_line, y_line)

    for color, (x_line, y_line) in line_dict.items():
        if color == 'red':
            sentiment_label = 'Very Negative'
        elif color == 'orange':
            sentiment_label = 'Negative'
        elif color == 'yellow':
            sentiment_label = 'Neutral'
        elif color == 'green':
            sentiment_label = 'Positive'
        elif color == 'blue':
            sentiment_label = 'Very Positive'

        plt.plot(x_line, y_line, color=color, linestyle='-', marker='o', label=f'Sentiment Level {sentiment_label}')

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1], loc='upper left', bbox_to_anchor=(1, 1))

    plt.title('Sentiment Scores Over Time with Key Exclusive Words (Scatter Plot)')
    plt.xlabel('Publication Date')
    plt.ylabel('Sentiment Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Define sentiment ranges 
sentiment_ranges = [(-1.0, -0.6), (-0.6, -0.2), (-0.2, 0.2), (0.2, 0.6), (0.6, 1.0)]

# Generate and display word clouds for each sentiment range
for sentiment_range in sentiment_ranges:
    generate_wordcloud_for_sentiment(sentiment_range)

# Create and display the scatter plot for sentiment scores over time
scatter_plot_sentiment_over_time()


def plot_sentiment_by_week(articles, sentiment_label, color):
    # Check if 'Week' is in the columns
    if 'Week' not in articles.columns:
        # Check if 'Publication Date' is in the columns
        if 'Publication Date' not in articles.columns:
            print("Error: 'Publication Date' column not found in the DataFrame.")
            return

        # Ensure 'Publication Date' is in datetime format
        if pd.api.types.is_datetime64_any_dtype(articles['Publication Date']):
            # Extract week from the 'Publication Date'
            articles['Week'] = articles['Publication Date'].dt.strftime('%Y-%U')
        else:
            print("Error: 'Publication Date' is not in datetime format.")
            return
    else:
        # Handle the case where 'Week' is already present
        pass

    # Count articles by week
    article_counts_by_week = articles['Week'].value_counts().sort_index()

    # Plotting the bar plot
    plt.figure(figsize=(15, 8))
    article_counts_by_week.plot(kind='bar', color=color, alpha=0.7)
    plt.title(f'Article Count for {sentiment_label} Articles by Week')
    plt.xlabel('Week')
    plt.ylabel('Article Count')
    plt.xticks(rotation=45, ha='right')  # rotate x-axis labels for better visibility
    plt.tight_layout()
    plt.show()

def display_selected_week_articles(articles, selected_week):
    # Extract articles for the specified week
    articles_for_selected_week = articles[articles['Week'] == selected_week]

    # Display the details of articles for the specified week along with the sentiment score
    print(f"\nArticles for Week {selected_week}")
    for idx, row in articles_for_selected_week.iterrows():
        print(f"Publication Date: {row['Publication Date']}, Sentiment Score: {row['Sentiment Score']:.4f}, Summary: {row['Summary']}") 

# Function to generate and display bar plots for 'Very Negative' and 'Very Positive' sentiments
def generate_sentiment_bar_plots(df):
    # Filter articles for 'Very Negative' sentiment range
    very_negative_articles = df[(df['Sentiment Score'] >= -1.0) & (df['Sentiment Score'] < -0.6)]
    plot_sentiment_by_week(very_negative_articles, 'Very Negative', 'red')
    selected_week_very_negative = very_negative_articles['Week'].value_counts().idxmax()
    display_selected_week_articles(very_negative_articles, selected_week_very_negative)

    # Filter articles for 'Very Positive' sentiment range
    very_positive_articles = df[(df['Sentiment Score'] >= 0.6) & (df['Sentiment Score'] <= 1.0)]
    plot_sentiment_by_week(very_positive_articles, 'Very Positive', 'blue')
    selected_week_very_positive = very_positive_articles['Week'].value_counts().idxmax()
    display_selected_week_articles(very_positive_articles, selected_week_very_positive)

# Example for Very Negative and Very Positive articles
generate_sentiment_bar_plots(df_processed_text)
```

    Articles contributing to Very Negative Sentiment:
    
                Publication Date  \
    0  2023-12-06 06:54:24+00:00   
    2  2023-12-08 16:22:02+00:00   
    4  2023-11-14 17:33:06+00:00   
    6  2023-11-21 11:00:13+00:00   
    7  2023-09-20 20:50:39+00:00   
    9  2023-12-07 17:07:49+00:00   
    10 2023-12-03 08:00:09+00:00   
    13 2023-12-06 20:27:34+00:00   
    15 2023-10-31 19:08:46+00:00   
    16 2023-06-13 18:08:16+00:00   
    18 2023-12-02 13:29:49+00:00   
    22 2023-12-11 16:58:01+00:00   
    24 2023-12-06 07:00:04+00:00   
    26 2023-10-31 09:24:58+00:00   
    28 2023-10-18 05:45:32+00:00   
    32 2023-11-30 17:30:15+00:00   
    34 2023-11-21 16:01:06+00:00   
    35 2023-12-20 15:40:03+00:00   
    36 2023-11-14 16:03:57+00:00   
    39 2023-12-09 09:00:16+00:00   
    40 2023-11-22 11:11:31+00:00   
    42 2023-11-26 07:00:13+00:00   
    43 2023-12-04 11:10:30+00:00   
    44 2023-11-03 08:00:04+00:00   
    45 2023-11-01 20:06:07+00:00   
    47 2023-11-29 18:35:33+00:00   
    48 2023-10-17 16:09:45+00:00   
    
                                                  Summary  
    0   Boris Johnson faces tough questions at Covid i...  
    2    ‘Completely out of touch’: five people hit ha...  
    4   Local expertise was ignored during Covid pande...  
    6   What does the Covid inquiry really tell us? If...  
    7   Morning Mail: AFL’s gambling problem, Covid-19...  
    9   Boris Johnson tells Covid inquiry he avoided e...  
    10   Covid inquiry: 10 questions facing Boris Johnson  
    13     Boris Johnson at the Covid inquiry: key points  
    15  The Guardian view on the Covid inquiry: shocki...  
    16  ‘A bowl of spaghetti’ : Covid inquiry opens wi...  
    18  Covid inquiry: Boris Johnson ‘to admit he made...  
    22  Boris Johnson’s negligence has been laid bare ...  
    24  Boris Johnson ‘trying to rewrite history’ befo...  
    26  Brexit prioritised over tackling Covid at star...  
    28  Wednesday briefing: What you’ve missed at the ...  
    32  Matt Hancock ‘was not told about eat out to he...  
    34  I regret saying ‘herd immunity’, Vallance tell...  
    35  Women’s voices barely heard in Boris Johnson’s...  
    36  Incarcerated students earn degrees in groundbr...  
    39  Eating out and Partygate: Covid inquiry questi...  
    40  Herd immunity was never UK policy, Chris Whitt...  
    42  Stretched NHS even less ready to cope with a n...  
    43  Key questions Boris Johnson is likely to face ...  
    44  Macho posturing costs lives: another lesson fr...  
    45  Dead cats and hairdryers: Dominic Cummings’ ev...  
    47  Rishi Sunak pushed hard for lifting of Covid r...  
    48  Communication with ministers was poor, scienti...  
    


    
![png](output_5_1.png)
    


    Articles contributing to Negative Sentiment:
    
                Publication Date  \
    8  2023-09-17 06:00:53+00:00   
    12 2023-09-21 07:06:21+00:00   
    21 2023-12-02 07:27:46+00:00   
    46 2023-06-01 18:00:37+00:00   
    49 2023-11-09 17:55:53+00:00   
    
                                                  Summary  
    8   Sunak and Johnson Covid-19 inquiry appearances...  
    12  Peak medical body labels Covid-19 review ‘half...  
    21  Boris Johnson’s legacy will be shaped by Covid...  
    46  The Guardian view on the Covid-19 standoff: it...  
    49  Yousaf apologises for ‘shortcomings’ with rele...  
    


    
![png](output_5_3.png)
    


    Articles contributing to Neutral Sentiment:
    
                Publication Date  \
    14 2023-06-14 14:53:25+00:00   
    25 2023-12-15 16:00:32+00:00   
    
                                                  Summary  
    14  Covid testing was a weakness in early pandemic...  
    25  Hugo Keith KC: dogged fact-finder at heart of ...  
    


    
![png](output_5_5.png)
    


    Articles contributing to Positive Sentiment:
    
                Publication Date  \
    11 2023-11-03 14:41:30+00:00   
    23 2023-12-07 17:29:17+00:00   
    31 2023-12-01 16:56:20+00:00   
    
                                                  Summary  
    11  Another truth from the Covid inquiry: women we...  
    23  Boris Johnson’s second day at the Covid inquir...  
    31  Boris Johnson wanted ‘massive fines’ for lockd...  
    


    
![png](output_5_7.png)
    


    Articles contributing to Very Positive Sentiment:
    
                Publication Date  \
    1  2023-10-04 17:11:55+00:00   
    3  2023-11-02 18:39:22+00:00   
    5  2023-09-21 01:13:11+00:00   
    17 2023-12-11 18:21:39+00:00   
    19 2023-10-06 08:51:48+00:00   
    20 2023-11-20 23:17:46+00:00   
    27 2023-11-01 14:03:38+00:00   
    29 2023-11-20 17:23:50+00:00   
    30 2023-11-27 16:27:57+00:00   
    33 2023-10-30 05:00:04+00:00   
    37 2023-11-02 19:29:17+00:00   
    38 2023-02-16 18:57:47+00:00   
    41 2023-06-13 05:43:33+00:00   
    
                                                  Summary  
    1   Children were failed by pandemic policies, Cov...  
    3   The Guardian view on the Covid-19 inquiry: a w...  
    5   Covid-19 inquiry will exclude state and territ...  
    17  ‘Deeply sorry’: what Rishi Sunak said to the C...  
    19  Wetherspoon’s returns to profit for first time...  
    20  Patrick Vallance contradicts Rishi Sunak’s evi...  
    27  ‘Absence of humanity’: Helen MacNamara’s evide...  
    29  What we learned from Patrick Vallance at the C...  
    30  UK Covid response was London-centric, Andy Bur...  
    33  UK Covid inquiry reaches week of potentially d...  
    37  What we learned about Matt Hancock from the Co...  
    38  PR firm behind Tory pandemic response linked t...  
    41  Tuesday briefing: A years-long inquiry into th...  
    


    
![png](output_5_9.png)
    


    C:\Users\lundt\AppData\Local\Temp\ipykernel_19264\3294544777.py:56: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      filtered_articles.sort_values(by='Publication Date', inplace=True)
    C:\Users\lundt\AppData\Local\Temp\ipykernel_19264\3294544777.py:56: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      filtered_articles.sort_values(by='Publication Date', inplace=True)
    C:\Users\lundt\AppData\Local\Temp\ipykernel_19264\3294544777.py:56: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      filtered_articles.sort_values(by='Publication Date', inplace=True)
    C:\Users\lundt\AppData\Local\Temp\ipykernel_19264\3294544777.py:56: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      filtered_articles.sort_values(by='Publication Date', inplace=True)
    C:\Users\lundt\AppData\Local\Temp\ipykernel_19264\3294544777.py:56: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      filtered_articles.sort_values(by='Publication Date', inplace=True)
    


    
![png](output_5_11.png)
    


    C:\Users\lundt\AppData\Local\Temp\ipykernel_19264\3294544777.py:119: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      articles['Week'] = articles['Publication Date'].dt.strftime('%Y-%U')
    


    
![png](output_5_13.png)
    


    
    Articles for Week 2023-49
    Publication Date: 2023-12-03 08:00:09+00:00, Sentiment Score: -0.9382, Summary: Covid inquiry: 10 questions facing Boris Johnson
    Publication Date: 2023-12-04 11:10:30+00:00, Sentiment Score: -0.9657, Summary: Key questions Boris Johnson is likely to face at Covid inquiry
    Publication Date: 2023-12-06 06:54:24+00:00, Sentiment Score: -0.8402, Summary: Boris Johnson faces tough questions at Covid inquiry over handling of pandemic 
    Publication Date: 2023-12-06 07:00:04+00:00, Sentiment Score: -0.9360, Summary: Boris Johnson ‘trying to rewrite history’ before Covid inquiry appearance
    Publication Date: 2023-12-06 20:27:34+00:00, Sentiment Score: -0.9966, Summary: Boris Johnson at the Covid inquiry: key points
    Publication Date: 2023-12-07 17:07:49+00:00, Sentiment Score: -0.9671, Summary: Boris Johnson tells Covid inquiry he avoided engaging with devolved administrations during pandemic for political reasons – as it happened
    Publication Date: 2023-12-08 16:22:02+00:00, Sentiment Score: -0.9349, Summary:  ‘Completely out of touch’: five people hit hard by pandemic on Johnson at Covid inquiry
    Publication Date: 2023-12-09 09:00:16+00:00, Sentiment Score: -0.9580, Summary: Eating out and Partygate: Covid inquiry questions Sunak should prepare for
    

    C:\Users\lundt\AppData\Local\Temp\ipykernel_19264\3294544777.py:119: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      articles['Week'] = articles['Publication Date'].dt.strftime('%Y-%U')
    


    
![png](output_5_16.png)
    


    
    Articles for Week 2023-44
    Publication Date: 2023-10-30 05:00:04+00:00, Sentiment Score: 0.6815, Summary: UK Covid inquiry reaches week of potentially damning testimony
    Publication Date: 2023-11-01 14:03:38+00:00, Sentiment Score: 0.9718, Summary: ‘Absence of humanity’: Helen MacNamara’s evidence to Covid inquiry
    Publication Date: 2023-11-02 18:39:22+00:00, Sentiment Score: 0.8360, Summary: The Guardian view on the Covid-19 inquiry: a week that has probed deep | Editorial
    Publication Date: 2023-11-02 19:29:17+00:00, Sentiment Score: 0.7506, Summary: What we learned about Matt Hancock from the Covid inquiry
    


```python
## Validation and model evaluation

# Save the entire code as a dictionary
code_to_pickle = {
    'code': '''
        # Create a Sentiment Intensity Analyzer for validation
vader_analyzer = SentimentIntensityAnalyzer()

# Create a list to store validation data
validation_data_list = []

# Iterate through articles for validation
for article in articles:
    pub_date = article["webPublicationDate"]
    summary = article["webTitle"]

    # Fetch the content from the article's URL
    article_url = article["webUrl"]
    article_response = requests.get(article_url)

    # Check the content type
    content_type = article_response.headers.get('content-type', '').lower()

    try:
        if 'application/json' in content_type:
            article_data = article_response.json()
            article_content = article_data.get("response", {}).get("content", {}).get("blocks", {}).get("body", "")
        else:
            article_content = article_response.text

        if article_content:
            # Perform text preprocessing with additional stopwords
            preprocessed_text = preprocess_text(article_content, additional_stopwords)

            # Perform sentiment analysis using the unsupervised model
            unsupervised_sentiment_score = sia.polarity_scores(preprocessed_text)['compound']

            # Perform VADER sentiment analysis for validation
            vader_sentiment_score = vader_analyzer.polarity_scores(article_content)['compound']

            # Convert publication date to datetime format
            pub_date = pd.to_datetime(pub_date)

            # Store data in a dictionary for validation
            validation_data_dict = {
                'Publication Date': pub_date,
                'Summary': summary,
                'Unsupervised Sentiment Score': unsupervised_sentiment_score,
                'VADER Sentiment Score': vader_sentiment_score
            }

            # Append the dictionary to the list
            validation_data_list.append(validation_data_dict)

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")

# Create a data frame for validation
df_validation = pd.DataFrame(validation_data_list)

# Merge validation and processed text dataframes
df_combined_validation = pd.merge(df_processed_text, df_validation, on=['Publication Date', 'Summary'])

# Print unsupervised sentiment score and VADER sentiment score side by side
print("Unsupervised vs. VADER Sentiment Scores:")
print(pd.DataFrame({
    'Unsupervised Sentiment Score': df_combined_validation['Unsupervised Sentiment Score'],
    'VADER Sentiment Score': df_combined_validation['VADER Sentiment Score']
}))

# Evaluate the accuracy of the unsupervised model
accuracy = accuracy_score(df_combined_validation['Sentiment Score'].apply(lambda x: 1 if x >= 0 else 0),
                           df_combined_validation['VADER Sentiment Score'].apply(lambda x: 1 if x >= 0 else 0))

print(f"\nAccuracy of the Unsupervised Model: {accuracy:.2%}")

# Print classification report for the unsupervised model
print("\nClassification Report for the Unsupervised Model:")
print(classification_report(df_combined_validation['Sentiment Score'].apply(lambda x: 1 if x >= 0 else 0),
                            df_combined_validation['VADER Sentiment Score'].apply(lambda x: 1 if x >= 0 else 0)))

    '''
}

```


```python
## Pickling and README creation

# Save the VADER model to a file
vader_model_filename = 'final_model_vader.pkl'
with open(vader_model_filename, 'wb') as vader_model_file:
    pickle.dump(vader_analyzer, vader_model_file)

print(f"VADER model saved to {vader_model_filename}")


# Specify the file path where you want to save the pickle file
pickle_file_path = 'vader_sentiment_analysis.pkl'

# Dump the code dictionary into a pickle file
with open(pickle_file_path, 'wb') as file:
    pickle.dump(code_to_pickle, file)

print(f'Code pickled and saved to {pickle_file_path}')

```

    VADER model saved to final_model_vader.pkl
    Code pickled and saved to vader_sentiment_analysis.pkl
    
