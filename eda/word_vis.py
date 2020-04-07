"""
Extracts word count from the messages using CountVectorizer. Outputs a csv file
with the most and least frequent word occurance throughout the documents.
"""
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import StemmerI, SnowballStemmer
from nltk.stem.porter import PorterStemmer

import re
import pandas as pd
import numpy as np

from sqlalchemy import create_engine

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF


#%%

def load_data(database_filepath):
    """
    Import data from database into a DataFrame. Split DataFrame into
    features and predictors, `X` and `Y`. Additionally, extract the names
    of target categories.

    Preprocess data.

    Params:
    -------
        database_filepath: file path of database

    Returns:
    -------
        tuple(X, Y, category_names)
        pd.DataFrame of features and predictors, `X` and `Y`, respectively.
        List of target category names
    """
    engine = create_engine(f'sqlite:///{database_filepath}')

    # extract directory name
    dir_ = re.findall(".*/", database_filepath)

    # extract table name by stripping away directory name
    table_name = database_filepath.replace('.db', '').replace(dir_[0], "")

    df = pd.read_sql_table(f'{table_name}', engine)

    # reset index
    df.reset_index(drop=False, inplace=True)

    # DROP ROWS/COLUMN
    # where sum across entire row is less than 1
    null_idx = np.where(df.loc[:, 'related':].sum(axis=1) < 1)[0]
    # drop rows which contain all null values
    df.drop(null_idx, axis=0, inplace=True)

    # explore `related` feature where its labeled as a `2`
    related_twos = df[df['related'] == 2]
    df.drop(index=related_twos.index, inplace=True)

    # reset index
    df = df.reset_index(drop=True)

    # define features and predictors
    X = df.loc[:, 'message']
    Y = df.loc[:, 'related':]

    # drop categories with less than 2 classes
    drop_catg_list = Y.nunique()[Y.nunique() < 2].index.tolist()
    df.drop(drop_catg_list, axis=1, inplace=True)

    # extract label names
    category_names = Y.columns.to_list()

    return X, Y, df, category_names

def tokenize(text):
    """
    Replace `url` with empty space "".
    Tokenize and lemmatize input `text`.
    Converts to lower case and strips whitespaces.


    Returns:
    --------
        dtype: list, containing processed words
    """

    lemm = WordNetLemmatizer()
    stemmer = SnowballStemmer('english')

    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "")

    # load stopwords
    stop_words = stopwords.words("english")

    remove_words = ['one', 'see', 'please', 'thank', 'thank you', 'thanks',
                    'we', 'us', 'you', 'me', 'their', 'there', 'here', 'http']
    for addtl_word in remove_words:
        stop_words.append(addtl_word)

    # remove punctuations (retain alphabetical and numeric chars) and convert to all lower case
    # tokenize resulting text
    tokens = word_tokenize(re.sub(r"[^a-zA-Z]", ' ', text.lower().strip()))

    # drop stop words
    no_stops = [word for word in tokens if word not in stop_words]

    # lemmatize and remove stop words
    lemmatized = [lemm.lemmatize(word) for word in tokens if word not in stop_words]
    # stemmed = [stemmer.stem(word) for word in tokens if word not in stop_words]


    return lemmatized

#%%

X, Y, df, category_names = load_data('../data/disaster_response.db')


#%%
count_vec = CountVectorizer(
        tokenizer=tokenize,
        ngram_range=(1, 1),
        dtype=np.uint16,
        max_features=10000,
        max_df=0.99,
        min_df=2,
        )


word_matrix = count_vec.fit_transform(X).toarray()

# extract words/features
num_features = len(count_vec.get_feature_names())
print('-'*75)
print('Number of features:', num_features)
features = count_vec.get_feature_names()

# sum counts of words; axis=0 for column wise summation
word_count = np.sum(word_matrix, axis=0)


# COUNTVECTORIZER VOCABULARY
# create dataframe with results
df = pd.DataFrame({'word': features,
                   'count': word_count})

### Unigram
bottom_words = df.sort_values('count', ascending=True)[:25]
top_words = df.sort_values('count', ascending=True)[-25:]

top_words.to_csv('../data/top_words.csv')
bottom_words.to_csv('../data/bottom_words.csv')


#### Bigram
# bigram_bottom_words = df.sort_values('count', ascending=True)[:25]
# bigram_top_words = df.sort_values('count', ascending=True)[-25:]

# bigram_top_words.to_csv('../data/bigram_top_words.csv')
# bigram_bottom_words.to_csv('../data/bigram_bottom_words.csv')


#%%
# def print_top_words(model, feature_names, n_top_words):
#     for topic_idx, topic in enumerate(model.components_):
#         message = "Topic #%d: " % topic_idx
#         message += " ".join([feature_names[i]
#                              for i in topic.argsort()[:-n_top_words - 1:-1]])
#         print(message)
#     print()

# nmf = NMF(n_components=10,
#           random_state=11,
#           alpha=.1,
#           l1_ratio=.5).fit(word_matrix)

# components = nmf.components_
# components[:-5 - 1:-1]

# # print topic
# print_top_words(nmf, features, 12)


# " ".join([features[i] for i in components.argsort()[:-8 - 1:-1]])


# ls = [1,2,4,6,7,8,34,4]
# ls[:-5-1:-1]
# ls[:-5:-1]








