# Feature-Based Sentiment Analysis

Feature-Based Sentiment Analysis, also known as [Aspect-Based Sentiment Analysis](https://monkeylearn.com/blog/aspect-based-sentiment-analysis/), is an advanced text analysis technique of customer feedback data to learn more about your customers and/or your competitors' customers. This analysis tells you which features people are talking the most and if the sentiment towards each feature is positive or negative. It is a powerful tool for Product Managers, Product Marketing Managers, and entrepreneurs.

## How does this technique help? 

1) Analyze massive amounts of data in detail, saving money and time.
2) It's a great tool to help you decide on your MVP features and build your product roadmap. 
3) Also helpful for feature prioritization and product strategy. 
4) Learn how people are responding to a new important update or new feature release.
5) It will help you build your product competitive matrix and segment customers.
6) Learn how a new marketing campaign is resonating with your target customers.
7) Help you determine the marketing strategy of a new product launch.
8) Also helpful to decide how to position your product to different segments.
9) Learn about a new market entry opportunity.

## About this project

This project aims to analyze customer reviews of Alexa products using NLP and ML in Python, but it can be used to analyze other sources like social media posts, user reviews on Play Store & App Store, and internal customer feedback data like CSAT or NPS responses.
I got the dataset from [Kaggle](https://www.kaggle.com/sid321axn/amazon-alexa-reviews) and did some changes so it is more versatile and you can use it for multiple products/apps. The dataset is on reviews.csv file and after running absa.py the program will create an Excel file called Alexa.xlsx. 
We are using a small dataset of 3.150 reviews but you can use a much larger one.

## How to run the analysis

1) Install all files that are in demo_gh folder to your project's folder: stop_words_list.py, absa.py, requirements.txt, and reviews.csv. Do not install Alexa.xlsx to your project folder because I provided this file only so you can see how the output file should look like.
2) Go to your terminal and type: pip install -r requirements.txt
This will install all necessary libraries such as nltk (natural language toolkit) and scikit-learn (ML library). Make sure you are in a virtual environment before installing the requirement packages. If you don't understand what a virtual environment is, read the explanation at the end of this file.
3) Check that your data structure is similar to the one provided in this project called reviews.csv. If not, make necessary changes.
4) Open the absa.py file in your IDE and follow the steps. You will need to complete the settings to match your data, which are app_name, aspects_no, and reviews_name_col. 
5) Run the absa.py file typing in your terminal: py absa.py
6) Go to your folder and open the Excel file with the name of your product/app to get insights from your analysis. Make sure you assign correctly the feature keywords with the dominant topics extracted with this technique (sheet name Topics_key).

## How to get powerful insights from the results

Go from a high-level first analysis to a more detailed analysis.
1) Create a pivot table and dynamic bar chart showing number of positive, negative, and neutral sentiment reviews per feature. 
2) See if there are any surprising features you weren't expecting to find.
3) Go to the Reviews sheet and filter by the features that are most important and that called your attention. Filter out the neutral sentiment reviews to focus on positive and negative reviews.

## References

1) [A Comprehensive Guide to Aspect-based Sentiment Analysis](https://monkeylearn.com/blog/aspect-based-sentiment-analysis/)
2) [25 Best NLP Datasets for Machine Learning Projects](https://lionbridge.ai/datasets/the-best-25-datasets-for-natural-language-processing/)
3) [Implementing Aspect Based Sentiment Analysis using Python](https://medium.com/analytics-vidhya/aspect-based-sentiment-analysis-a-practical-approach-8f51029bbc4a)
4) [Lowri Williams' GitHub](https://github.com/LowriWilliams/Aspect_Sentiment_Analysis)
5) [Sentiment Analysis: Aspect-Based Opinion Mining](https://towardsdatascience.com/%EF%B8%8F-sentiment-analysis-aspect-based-opinion-mining-72a75e8c8a6d) 
6) [Rule-based Sentiment Analysis of App Store Review in Python](https://towardsdatascience.com/rule-based-sentiment-analysis-of-app-store-review-in-python-94d8bbfc48bb)

## Why do we need to create a virtual environment

The virtual environment is something you create when you start a project. It's always advisable to create a virtual environment for your new project so the libraries you install there don't conflict with your other projects that might demand other versions of the libraries. For example, I am using vaderSentiment version 3.3.2 for this project, but for another project, I might need vaderSentiment version 3.2.5.

To create a virtual environment in Python 3 and using VS Code as your IDE, write this in the terminal:
py -3 -m venv name_of_project
And to activate the virtual environment type
name_of_project\Scripts\activate
