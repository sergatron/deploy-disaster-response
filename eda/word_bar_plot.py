# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 20:36:45 2020

@author: smouz

"""

from plotly.graph_objs import Figure, Bar



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
    fig.update_layout(title=title,
                      yaxis_title='Word',
                      plot_bgcolor='white',
                      yaxis=xy_axis,
                      xaxis=x_axis,

                     )
    return fig




