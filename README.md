# Identifying Mental Health Crises in r/OCD Posts

## Introduction
This research project aims to use Natural Language Processing (NLP) to detect crises within posts from the r/OCD subreddit. By understanding these patterns, we can potentially develop automated tools for early intervention and support.

Using this project, you can extract and analyze flair data from the r/OCD subreddit using TF-IDF and machine learning techniques. TF-IDF identifies the most informative words and bigrams in posts within each flair. The machine learning component evaluates models, including a baseline dummy classifier, Naive Bayes, SVM, and Decision Tree, to determine the best one for detecting crises.

## Usage

1. **Extract Posts:** Run <code>extract_posts.py</code> to fetch posts. You will need to enter your [Reddit API credentials](https://www.reddit.com/wiki/api/) and and adhere to the [Reddit API access rules](https://support.reddithelp.com/hc/en-us/articles/16160319875092-Reddit-Data-API-Wiki), which include restrictions on storing data. This will separate the data into "Crisis" and "Non-crisis" categories.
2. **Analyze Data:** After extracting posts, you can run the TF-IDF and machine learning analysis scripts. These scripts will print out the top 15 TF-IDF features per flare and the performace of the various classifiers, respectively.