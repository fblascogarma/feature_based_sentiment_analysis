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
df_reviews = pd.read_csv("./reviews.csv", encoding="utf-8")     # this file contains reviews I want to analyze

## Settings to match your data

app_name = "Alexa"                                                                                  # the name of the app or product you want to analyze
features = 5                                                                                        # number of features you want to analyze
reviews_name_col = 'verified_reviews'                                                               # name of the column that has the app reviews in your file
language_of_reviews = 'english'                                                                     # set the languague of your reviews (check list of 20 options down below)
language_of_reviews_list = {'english', 'spanish', 'portuguese', 'french', 'german',
                            'arabic', 'azerbaijani', 'danish', 'dutch', 'finnish', 'greek', 'hungarian',
                            'indonesian', 'italian', 'kazakh', 'nepali', 'norwegian', 'romanian', 'russian', 'slovene'}

## Start the aspect-based sentiment analysis

# 1. Create a df
df_reviews = df_reviews[df_reviews['App_Name']==app_name]
df_reviews = pd.DataFrame({
    'reviews': df_reviews[reviews_name_col]                           
})

# 2. Convert the text to lowercase and remove punctuation and white space
df_reviews['remove_lower_punct'] = df_reviews['reviews'].str.lower().str.replace("'", '', regex=True).str.replace('[^\w\s]', ' ', regex=True).str.replace(" \d+", " ", regex=True).str.replace(' +', ' ', regex=True).str.strip()

# 3. Apply sentiment analysis using VADER
analyser = SentimentIntensityAnalyzer()                         # this is a rule-based sentiment analysis tool for social media

sentiment_score_list = []
sentiment_label_list = []

for i in df_reviews['remove_lower_punct'].values.tolist():
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
    
df_reviews['sentiment'] = sentiment_label_list
df_reviews['sentiment score'] = sentiment_score_list

# 4. Tokenise string
df_reviews['tokenise'] = df_reviews.apply(lambda row: nltk.word_tokenize(row[1]), axis=1)         # <apply> function to tokenize and <lambda> to find length of each text

# 5. Initiate stopwords from nltk, add additional missing terms, and remove stopwords
stop_words = stopwords.words(language_of_reviews)                       # there are 21 languages to use and you can check the options going to C:/Users/username/AppData/Roming/nltk_data/corpora/stopwords
stop_words.extend(stop_words_list)                                      # to increase the stop_words list with the words we included in our file with that name
df_reviews['remove_stopwords'] = df_reviews['tokenise'].apply(lambda x: [item for item in x if item not in stop_words])

# 6. Initiate nltk Lemmatiser and Lemmatise words
wordnet_lemmatizer = WordNetLemmatizer()
df_reviews['lemmatise'] = df_reviews['remove_stopwords'].apply(lambda x: [wordnet_lemmatizer.lemmatize(y) for y in x])

# 7. Initialize the count vectorizer and join the processed data to be a vectorised
vectorizer = CountVectorizer(analyzer = 'word', ngram_range = (2, 2))       # convert a collection of text documents to a matrix of token counts
vectors = []
for index, row in df_reviews.iterrows():
    vectors.append(", ".join(row[6]))
vectorised = vectorizer.fit_transform(vectors)

# 8. Initisalize LDA model and make the df
lda_model = LatentDirichletAllocation(n_components = features,            # number of topics; default = 10
                                    random_state = 10,
                                    evaluate_every = -1,                    # compute perplexity every n iters you want, but for practicality you don't need to as doing so will increase the training time
                                    n_jobs = -1,                            # use all available CPUs
                                    )

lda_output = lda_model.fit_transform(vectorised)

topic_names = ["Topic" + str(i) for i in range(1, lda_model.n_components + 1)]      # col names

df_reviews_document_topic = pd.DataFrame(np.round(lda_output, 2), columns = topic_names)

# 9. Get dominant topic for each document and join to original df
dominant_topic = (np.argmax(df_reviews_document_topic.values, axis=1)+1)
df_reviews_document_topic['Dominant_topic'] = dominant_topic

df_reviews = df_reviews.reset_index()             # reset index to do a proper merge between the 2 df
df_reviews = pd.merge(df_reviews, df_reviews_document_topic, left_index = True, right_index = True, how = 'outer')

# 10. Keywords the LDA extracted from such reviews
docnames = ['Doc' + str(i) for i in range(len(df_reviews))]                     # index names (one doc per row)
df_reviews_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topic_names, index=docnames)             # make the df of docs


dominant_topic = np.argmax(df_reviews_document_topic.values, axis=1)           # get dominant topic position for each document (0 is Topic1, 1 is Topic2, and so on)
df_reviews_document_topic['dominant_topic'] = dominant_topic

df_reviews_topic_keywords = pd.DataFrame(lda_model.components_)                # topic-keyword Matrix


df_reviews_topic_keywords.columns = vectorizer.get_feature_names()             # assign Column and Index
df_reviews_topic_keywords.index = topic_names

df_topic_no = pd.DataFrame(df_reviews_topic_keywords.idxmax())
df_scores = pd.DataFrame(df_reviews_topic_keywords.max())

tmp = pd.merge(df_topic_no, df_scores, left_index=True, right_index=True)
tmp.columns = ['topic', 'relevance_score']

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
all_topics['Dominant_topic'] = pd.to_numeric(all_topics['Dominant_topic'].str[5])
all_topics['topic_name'] = all_topics['topic_name'].astype(str)                     # convert the list into a string type to create a pivot table later
all_topics['topic_name'] = all_topics['topic_name'].str.split('[').str[1]
all_topics['topic_name'] = all_topics['topic_name'].str.split(']').str[0]           # this and previous line is to get the string between the [ ] symbols

# 12. Results
results = df_reviews.groupby(['Dominant_topic', 'sentiment']).count().reset_index()
results = pd.merge(results, all_topics, on='Dominant_topic', how='inner')

reviews = pd.merge(df_reviews, all_topics, on='Dominant_topic', how='inner')
reviews.rename(columns = {'topic_name':'aspect'}, inplace = True)
reviews = reviews[['reviews', 'sentiment', 'sentiment score', 'aspect']]

# table in nominal values
aspect_based_table_val = pd.crosstab(results.sentiment,                 # index
                                results.topic_name,                     # columns
                                values = results.reviews,
                                aggfunc = np.sum,
                                margins = True,
                                margins_name = 'Total')

# table in percentage values
aspect_based_table_pct = pd.crosstab(results.sentiment,                 # index
                                results.topic_name,                     # columns
                                values = results.reviews,
                                aggfunc = np.sum,
                                margins = True,
                                margins_name = 'Total',
                                normalize = 'columns')                  # to calculate the percentage over each column

# 13. Prepare data and export data to an Excel file

writer = pd.ExcelWriter(app_name+'.xlsx', engine='xlsxwriter')      # create a pandas Excel writer using XlsxWriter as the engine
workbook = writer.book                                              # access the XlsxWriter workbook and worksheet objects from the dataframe
format1 = workbook.add_format({'num_format': '#,##0'})              # format numbers for df output
format2 = workbook.add_format({'num_format': '0%'})                 # format percentages for df output
center = workbook.add_format({'align': 'center'})                   # format cells to be aligned to the center

worksheet_index = workbook.add_worksheet('Index')                   # this sheet is to help the user navigate the output file
worksheet_index.hide_gridlines(option=2)

aspect_based_table_pct.to_excel(writer, sheet_name = 'Analysis', startrow = 1, startcol = 1)        # data in percentages
aspect_based_table_val.to_excel(writer, sheet_name = 'Analysis', startrow = 6, startcol = 1)        # data in nominal values
worksheet_absa = writer.sheets['Analysis']
worksheet_absa.set_column('B:J', 13, center)
worksheet_absa.conditional_format('C8:J11', {
    'type': 'cell',
    'criteria': 'greater than or equal to',
    'value':    0,
    'format': format1
})
worksheet_absa.conditional_format('C3:J5', {
    'type': 'cell',
    'criteria': 'greater than or equal to',
    'value':    0,
    'format': format2
})

chart_sent = workbook.add_chart({'type': 'column'})             # chart showing sentiment analysis results
chart_sent.add_series({
    'categories': ['Analysis', 2, 1, 4, 1],                     # [sheetname, first_row, first_col, last_row, last_col]
    'values': ['Analysis', 2, features+2, 4, features+2],       # using features variable so it changes depending on how many features the user selects
    'data_labels': {'value': True, 'num_format': '0%'},
    'points': [
        {'fill': {'color': '#FF6969'}},                         # light red
        {'fill': {'color': '#4F81BD'}},                         # blue accent 1
        {'fill': {'color': '#9BBB59'}},                         # light olive green
        ],
    'gap': 20,
})
chart_sent.set_title({'name': 'Sentiment Analysis (%)'})                     
chart_sent.set_legend({'none': True})
chart_sent.set_x_axis({'major_gridlines': {'visible': False},})
chart_sent.set_y_axis({'major_gridlines': {'visible': False}, 'visible': False})
worksheet_absa.insert_chart('B13', chart_sent)

chart_absa = workbook.add_chart({'type': 'column'})             # chart showing sentiment analysis results
# create 3 series for each sentiment
chart_absa.add_series({
    'categories': ['Analysis', 1, 2, 1, 1+features],                     # [sheetname, first_row, first_col, last_row, last_col] # aspect
    'values': ['Analysis', 2, 2, 2, 1+features],
    'name': 'Negative',
    'data_labels': {'value': True, 'num_format': '0%'},
    'fill': {'color': '#FF6969'},                         # light red
    'gap': 20,
})
chart_absa.add_series({
    'categories': ['Analysis', 1, 2, 1, 1+features],                     
    'values': ['Analysis', 3, 2, 3, 1+features],
    'name': 'Neutral',
    'data_labels': {'value': True, 'num_format': '0%'},
    'fill': {'color': '#4F81BD'},                         # blue accent 1
    'gap': 20,
})
chart_absa.add_series({
    'categories': ['Analysis', 1, 2, 1, 1+features],                     
    'values': ['Analysis', 4, 2, 4, 1+features],
    'name': 'Positive',
    'data_labels': {'value': True, 'num_format': '0%'},
    'fill': {'color': '#9BBB59'},                         # light olive green
    'gap': 20,
})

chart_absa.set_title({'name': 'Aspect-Based Sentiment Analysis (%)'})                     
chart_absa.set_legend({'position': 'bottom'})
chart_absa.set_x_axis({'major_gridlines': {'visible': False},})
chart_absa.set_y_axis({'major_gridlines': {'visible': False}, 'visible': False})
chart_absa.set_size({'width': 1080, 'height': 288})
worksheet_absa.insert_chart('B28', chart_absa)

text_for_index = """
Dear data-driven friend,\n 
Thank you for using our app! Here are some things to help you maximize your insights.\n 
The {} most popular aspects of your product are: {}.\n 
In the Analysis sheet, you can see the sentiment of your users towards your product and specifically, towards aspects of your product. Sentiments are feelings that include emotions, attitudes, and opinions about your product. \n 
The 2 tables you will find there show the sentiment towards each aspect and to the product as a total, first in relative values (percentages) and then in nominal values. You will also find 2 charts to visualize this information. \n 
In the last sheet called Reviews, you have the sentiment score assigned to each of the reviews in case you want to go deeper in the analysis of a specific aspect of your product. \n 
Have fun!
""".format(features, all_topics['topic_name'].values.tolist())
options_for_textbox = {
    'width': 655,
    'height': 570,
    'x_offset': 5,
    'y_offset': 5,

    'font': {'color': 'black',
             'size': 14},
    'align': {'horizontal': 'left'},
    'gradient': {'colors': ['#DDEBCF',
                            '#9CB86E']},
}

worksheet_index.insert_textbox('B1', text_for_index, options_for_textbox)
                 
reviews.to_excel(writer, sheet_name = 'Reviews')
worksheet_reviews = writer.sheets['Reviews']
worksheet_reviews.set_column('C:E', 13, center)
worksheet_reviews.set_column('B:B', 13)
writer.save()                                                       # close the pandas Excel writer and output the Excel file
