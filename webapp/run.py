from webapp import app

import os
import sys

import re
import json
import plotly
import pandas as pd
import numpy as np

from plotly.graph_objs import Bar

from joblib import load
from sqlalchemy import create_engine

from flask import render_template, request

from models.custom_transform import tokenize
from eda.word_bar_plot import plot_bar

# add path to `models` directory to load `custom_transform`
wdir = os.getcwd()
print('\nWorking dir:', wdir)
# append path to `model` to load custom transformers
new_path = os.path.join(wdir, 'models')
if new_path not in sys.path:
    print("\nAdded new path:")
    sys.path.append(new_path)
    print()
    print(sys.path[-1])



#%%
# NOTE:
#   function `tokenize` should be imported from another script when training
#   the model initially.


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

#%%

# load model
try:
    model = load('models/clf_model.pkl')
except ModuleNotFoundError:
    print('Loaded model on 2nd attempt')
    model = load('clf_model.pkl')

X, Y, df, category_names = load_data('data/disaster_response.db')

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """
    Renders graphs created from the loaded data.
    Renders homepage with visualizations of the data.

    """
    # extract data needed for visuals
    genre_counts = df['genre'].value_counts().values
    genre_names = df['genre'].value_counts().index.to_list()

    # extract category names and count of each
    category_names = list(Y.sum().sort_values(ascending=False).index)
    category_values = list(Y.sum().sort_values(ascending=False))
    marker = dict(
        color='rgb(158,202,225)',
        line_color='rgb(8,48,107)',
        line_width=1.5,
        opacity=0.7
        )

    # create initial visuals
    graphs = [
        {
            'data': [
                Bar(x = genre_names,
                    y = genre_counts,
                    marker = marker)
                ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Genre"}
                }
            },
        {
            'data': [
                Bar(x = category_names,
                    y = category_values,
                    marker = marker)
                ],
            'layout': {
                'title': 'Target Category Count',
                'yaxis': {'title': 'Count',
                          'type': 'linear'
                          },
                'xaxis': {'title': 'Category',
                          'tickangle': -45,
                          }
                }
            },
    ]

    # load data
    top_words = pd.read_csv('data/top_words.csv')
    bottom_words = pd.read_csv('data/bottom_words.csv')
    bigram_top_words = pd.read_csv('data/bigram_top_words.csv')
    bigram_bottom_words = pd.read_csv('data/bigram_bottom_words.csv')

    #### Uni-gram Plots
    top_fig = plot_bar(x=top_words['count'],
                       y=top_words['word'],
                       title='Most Frequent Words (Unigram)')
    bottom_fig = plot_bar(x=bottom_words['count'],
                          y=bottom_words['word'],
                          title='Least Frequent Words (Unigram)')


    #### Bi-gram plots
    bigram_top_fig = plot_bar(
        x=bigram_top_words['count'],
        y=bigram_top_words['word'],
        title='Most Frequent Words (Bigram)')
    bigram_bottom_fig = plot_bar(
        x=bigram_bottom_words['count'],
        y=bigram_bottom_words['word'],
        title='Least Frequent Words (Bigram)')

    # convert plots to json; and append to list
    graphs.append(top_fig.to_plotly_json())
    graphs.append(bottom_fig.to_plotly_json())
    graphs.append(bigram_top_fig.to_plotly_json())
    graphs.append(bigram_bottom_fig.to_plotly_json())

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    """
    Renders a page which takes in user's query then passes
    the query to the model which makes predictions and outputs
    the labels to screen.
    """
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

