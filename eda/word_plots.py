
"""
Plotly graphs for use in the `run.py` script.

"""

from plotly.graph_objs import Figure, Bar, Scatter
import plotly.express as px
import numpy as np

#%%
def plot_bar(x, y, title=''):
    # define axis params to re-use
    xy_axis = dict(
        gridcolor='rgb(225, 225, 225)',
        gridwidth=0.25,
        linecolor='rgb(100, 100, 100)',
        linewidth=2,
        showticklabels=True,
        color='black'
    )
    # update x-axis params
    x_axis = xy_axis.copy()
    x_axis.update(dict(
        ticks='outside',
        tickfont=dict(
            family='Arial',
            color='rgb(82, 82, 82)',))
        )

    # Use the hovertext kw argument for hover text
    fig = Figure([
        Bar(x=x, y=y, orientation='h')
    ])

    # Customize aspect
    fig.update_traces(marker_color='rgb(158,202,225)',
                      marker_line_color='rgb(8,48,107)',
                      marker_line_width=1.5,
                      opacity=0.7)
    # Edit layout
    fig.update_layout(height=750,
                      title=title,
                      yaxis_title='Word',
                      plot_bgcolor='white',
                      yaxis=xy_axis,
                      xaxis=x_axis,)
    return fig


def plot_clusters():

    X_tsne = np.load('data/X_tsne.npy')
    y_pred = np.load('data/y_pred_clusters.npy')
    text_labels = np.load('data/text_labels.npy', allow_pickle=True)

    disc_colors = px.colors.qualitative.Light24

    # define y-axis properties
    y_axis = dict(
        gridcolor='rgb(225, 225, 225)',
        gridwidth=0.25,
        linecolor='rgb(100, 100, 100)',
        linewidth=2,
        showticklabels=True,
        color='black'
    )
    # update x-axis params
    x_axis = y_axis.copy()
    x_axis.update(dict(
        ticks='outside',
        tickfont=dict(
            family='Arial',
            color='rgb(82, 82, 82)',))
        )
    # Scatter figure
    fig = Figure([
        Scatter(
            x=X_tsne[:, 0],
            y=X_tsne[:, 1],
            mode='markers',
            marker_color=y_pred,
            marker=dict(size=6,
                        opacity=0.6,
                        colorscale=disc_colors),
            text=text_labels,
        )])
    # Edit layout
    fig.update_layout(
        title='Target Category Clusters (w/t-SNE & k-Means)',
        height=750,
        plot_bgcolor='white',
        yaxis=y_axis,
        xaxis=x_axis,
    #     showlegend=True,
        xaxis_title='Component 1',
        yaxis_title='Component 2',
    )

    return fig


