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


# add path to `models` directory to load `custom_transform`
wdir = os.getcwd()
# append path to `model` to load custom transformers
sys.path.append(wdir+'\models')

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

print("Trying to load model")
model = load('models/clf_model.pkl')

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

    # create visuals
    graphs = [
        {
            'data': [
                Bar(x = genre_names,
                    y = genre_counts)
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
                    y = category_values)
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

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

