import pandas as pd
import numpy as np
import nltk                                                             # natural language toolkit (suite of libraries and programs for NLP)
from nltk.corpus import stopwords, sentiwordnet as swn
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer             # ML library
from sklearn.decomposition import LatentDirichletAllocation
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer    # rule-based sentiment analysis tool
from stop_words_list import stop_words_list
import xlsxwriter

## Read data
df_reviews = pd.read_csv("./reviews.csv", encoding="utf-8")     # this file contains reviews for all my major competitors (used SearchMan API)

## Settings to match your data

app_name = "Alexa"                                                                                  # the name of the app or product you want to analyze
aspects_no = 5                                                                                      # number of features you want the algorithm to extract
reviews_name_col = 'verified_reviews'                                                               # name of the column that has the app reviews in your file
language_of_reviews = 'english'                                                                     # set the languague of your reviews (check list of 21 options down below)
language_of_reviews_list = {'english', 'spanish', 'portuguese', 'french', 'german',
                            'arabic', 'azerbaijani', 'danish', 'dutch', 'finnish', 'greek', 'hungarian',
                            'indonesian', 'italian', 'kazakh', 'nepali', 'norwegian', 'romanian', 'russian', 'slovene'}

## Start the aspect-based sentiment analysis

# 1. Create a df
df_reelgood = df_reviews[df_reviews['App_Name']==app_name]
df_reelgood = pd.DataFrame({
    'reviews': df_reelgood[reviews_name_col]                           
})

# 2. Convert the text to lowercase and remove punctuation and white space
df_reelgood['remove_lower_punct'] = df_reelgood['reviews'].str.lower().str.replace("'", '', regex=True).str.replace('[^\w\s]', ' ', regex=True).str.replace(" \d+", " ", regex=True).str.replace(' +', ' ', regex=True).str.strip()
#print(df_reelgood.head())
#print(len(df_reelgood))

# 3. Apply sentiment analysis using VADER
analyser = SentimentIntensityAnalyzer()                         # this is a rule-based sentiment analysis tool for social media

sentiment_score_list = []
sentiment_label_list = []

for i in df_reelgood['remove_lower_punct'].values.tolist():
    sentiment_score = analyser.polarity_scores(i)

    if sentiment_score['compound'] >= 0.05:
        sentiment_score_list.append(sentiment_score['compound'])
        sentiment_label_list.append('Positive')
    elif sentiment_score['compound'] > -0.05 and sentiment_score['compound'] < 0.05:
        sentiment_score_list.append(sentiment_score['compound'])
        sentiment_label_list.append('Neutral')
    elif sentiment_score['compound'] <= -0.05:
        sentiment_score_list.append(sentiment_score['compound'])
        sentiment_label_list.append('Negative')
    
df_reelgood['sentiment'] = sentiment_label_list
df_reelgood['sentiment score'] = sentiment_score_list
#print(df_reelgood.head())

# 4. Tokenise string
df_reelgood['tokenise'] = df_reelgood.apply(lambda row: nltk.word_tokenize(row[1]), axis=1)         # <apply> function to tokenize and <lambda> to find length of each text
#print(df_reelgood.head())

# 5. Initiate stopwords from nltk, add additional missing terms, and remove stopwords
stop_words = stopwords.words(language_of_reviews)                       # there are 21 languages to use and you can check the options going to C:/Users/username/AppData/Roming/nltk_data/corpora/stopwords
stop_words.extend(stop_words_list)                                      # to increase the stop_words list with the words we included in our file with that name
df_reelgood['remove_stopwords'] = df_reelgood['tokenise'].apply(lambda x: [item for item in x if item not in stop_words])
#print("step 5: ", df_reelgood.head())

# 6. Initiate nltk Lemmatiser and Lemmatise words
wordnet_lemmatizer = WordNetLemmatizer()
df_reelgood['lemmatise'] = df_reelgood['remove_stopwords'].apply(lambda x: [wordnet_lemmatizer.lemmatize(y) for y in x])
#print(df_reelgood.head())

# 7. Initialize the count vectorizer and join the processed data to be a vectorised
vectorizer = CountVectorizer(analyzer = 'word', ngram_range = (2, 2))       # convert a collection of text documents to a matrix of token counts
vectors = []
for index, row in df_reelgood.iterrows():
    vectors.append(", ".join(row[6]))
vectorised = vectorizer.fit_transform(vectors)
#print(vectorised)

# 8. Initisalize LDA model and make the df
lda_model = LatentDirichletAllocation(n_components = aspects_no,            # number of topics; default = 10
                                    random_state = 10,
                                    evaluate_every = -1,                    # compute perplexity every n iters you want, but for practicality you don't need to as doing so will increase the training time
                                    n_jobs = -1,                            # use all available CPUs
                                    )

lda_output = lda_model.fit_transform(vectorised)

topic_names = ["Topic" + str(i) for i in range(1, lda_model.n_components + 1)]      # col names

df_reelgood_document_topic = pd.DataFrame(np.round(lda_output, 2), columns = topic_names)
#print("df_reelgood_document_topic: ", df_reelgood_document_topic)

# 9. Get dominant topic for each document and join to original df
dominant_topic = (np.argmax(df_reelgood_document_topic.values, axis=1)+1)
df_reelgood_document_topic['Dominant_topic'] = dominant_topic
# print("Step 9 doc topic: ", df_reelgood_document_topic)
# print("Step 9 df pre-merge: ", df_reelgood.reset_index())

df_reelgood = df_reelgood.reset_index()             # reset index to do a proper merge between the 2 df
df_reelgood = pd.merge(df_reelgood, df_reelgood_document_topic, left_index = True, right_index = True, how = 'outer')
# print("Step 9 df: ", df_reelgood)

# 10. Keywords the LDA extracted from such reviews
docnames = ['Doc' + str(i) for i in range(len(df_reviews[df_reviews['App_Name']==app_name]))]                     # index names (one doc per row)
df_reelgood_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topic_names, index=docnames)             # make the df of docs
#print(df_reelgood_document_topic)


dominant_topic = np.argmax(df_reelgood_document_topic.values, axis=1)           # get dominant topic position for each document (0 is Topic1, 1 is Topic2, and so on)
df_reelgood_document_topic['dominant_topic'] = dominant_topic
#print(df_reelgood_document_topic)

df_reelgood_topic_keywords = pd.DataFrame(lda_model.components_)                # topic-keyword Matrix
#print(df_reelgood_topic_keywords)


df_reelgood_topic_keywords.columns = vectorizer.get_feature_names()             # assign Column and Index
df_reelgood_topic_keywords.index = topic_names

df_topic_no = pd.DataFrame(df_reelgood_topic_keywords.idxmax())
#print(df_topic_no)
df_scores = pd.DataFrame(df_reelgood_topic_keywords.max())

tmp = pd.merge(df_topic_no, df_scores, left_index=True, right_index=True)
tmp.columns = ['topic', 'relevance_score']
#print(tmp)

# 11. Determine which aspect a keyword belongs to, order df in descending order, and select the one with the highest score
all_topics = []

for i in tmp['topic'].unique():    
    tmp_1 = tmp.loc[tmp['topic'] == i].reset_index()
    tmp_1 = tmp_1.sort_values('relevance_score', ascending=False).head(1)

    tmp_1['topic'] = tmp_1['topic']
    
    tmp_2 = []
    tmp_2.append(tmp_1['topic'].unique()[0])
    tmp_2.append(list(tmp_1['index'].unique()))
    all_topics.append(tmp_2)

all_topics = pd.DataFrame(all_topics, columns=['Dominant_topic', 'topic_name'])
#print("all_topics: ", all_topics)

# 12. Results
#print("df_reelgood: ", df_reelgood)
#df_reelgood.to_csv("absa_reelgood.csv", encoding='utf8', index=False)
results = df_reelgood.groupby(['Dominant_topic', 'sentiment']).count().reset_index()
#print("results: ", results)

# 13. Export data to an Excel file

writer = pd.ExcelWriter(app_name+'.xlsx', engine='xlsxwriter')      # create a pandas Excel writer using XlsxWriter as the engine
results.to_excel(writer, sheet_name = 'Results')                    
df_reelgood.to_excel(writer, sheet_name = 'Reviews')
all_topics.to_excel(writer, sheet_name = 'Topics_key')
writer.save()                                                       # close the pandas Excel writer and output the Excel file
