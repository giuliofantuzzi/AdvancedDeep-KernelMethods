import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
#import plotly.graph_objects as go


def plot_2PC(projections, labels, labels_dict, color_map,title='Fashion MNIST (2 principal components)'):
    fig = px.scatter(x=projections[:,0], y=projections[:,1],
                    color=[labels_dict[label.item()] for label in labels],
                    color_discrete_map=color_map,
                    width=800, height=500)
    fig.update_traces(marker=dict(size=6, line=dict(color='black', width=1)))
    fig.update_layout(title=dict(text=title,
                                font=dict(color='black'),
                                x=0.5,
                                y = 0.95,
                                xanchor='center',
                                yanchor='top')
                    )
    fig.update_xaxes(title_text='PC 1')
    fig.update_yaxes(title_text='PC 2')
    return fig

def plot_3PC(projections,labels,labels_dict,color_map,title='Fashion MNIST (3 principal components)'):
    fig = px.scatter_3d(x=projections[:,0], y=projections[:,1], z=projections[:,2],
                        color=[labels_dict[label.item()] for label in labels],
                        color_discrete_map=color_map,
                        width=800, height=500)
    fig.update_traces(marker=dict(size=6, line=dict(color='black', width=1)))
    fig.update_layout(title=dict(text=title,
                                font=dict(color='black'),
                                x=0.5,
                                y = 0.95,
                                xanchor='center',
                                yanchor='top'))
    fig.update_scenes(xaxis_title_text='PC 1',
                      yaxis_title_text='PC 2',
                      zaxis_title_text='PC 3')
    return fig


def plot_Spectrum_and_CVR(kpca_eigenvalues,n_components_plot=None):
    if not n_components_plot:
        n_components_plot = kpca_eigenvalues.shape[0]
        
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.grid(zorder=1)
    ax1.plot(range(1, n_components_plot+1), kpca_eigenvalues[:n_components_plot],color='darkred',linestyle='--',linewidth=1,zorder=2)
    ax1.scatter(range(1, n_components_plot+1), kpca_eigenvalues[:n_components_plot],label='Singular values', s=30, color='indianred', edgecolors='black',zorder=2)
    ax1.set_xlabel('Eigenvalues Rank',fontsize=11)
    ax1.set_ylabel('Eigenvalues', color='darkred',fontsize=11)
    ax1.tick_params(axis='y', labelcolor='darkred')
    if n_components_plot:
        ax1.set_title(f'kPCA Spectrum and Cumulative Explained Variance Ratio (zoom on first {n_components_plot} PC)')
    else:
        ax1.set_title(f'kPCA Spectrum and Cumulative Explained Variance Ratio')
    ax2 = ax1.twinx()
    explained_variance_ratio = np.cumsum(kpca_eigenvalues) / np.sum(kpca_eigenvalues)
    ax2.plot(range(1, 101), explained_variance_ratio[:n_components_plot], label='Cumulative explained variance', linestyle='--', color='navy',linewidth=1,zorder=2)
    ax2.set_ylabel('Cumulative Explained Variance Ratio', color='navy',labelpad=10,fontsize=11)
    ax2.tick_params(axis='y', labelcolor='navy')
    fig.tight_layout()
    return fig