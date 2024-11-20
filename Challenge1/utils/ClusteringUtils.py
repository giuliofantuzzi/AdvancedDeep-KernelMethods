import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import torch as th

def plot_clusters_composition(cluster_composition_dict,labels_dict,color_map,
                              suptitle='Composition of clusters w.r.t. true labels'):
    fig,ax = plt.subplots(nrows=2,ncols=5,figsize=(55,20))

    for cluster_id, labels in cluster_composition_dict.items():
        unique, counts = th.unique(labels, return_counts=True)
        ax_ = ax[cluster_id//5, cluster_id%5]
        ax_.barh([labels_dict[label.item()] for label in unique],
                counts,
                color=[color_map[labels_dict[label.item()]] for label in unique],edgecolor='black')
        ax_.set_title(f'Cluster {cluster_id}',fontdict={'fontsize': 16,'fontweight':'bold'})
        for count in ax_.get_xticklabels():
            count.set_fontsize(12)
            count.set_fontweight('bold')
        for label in ax_.get_yticklabels():
            label.set_fontsize(14)
            label.set_fontweight('bold')
    #fig.tight_layout()
    fig.suptitle(suptitle,fontsize=20,fontweight='bold');
    plt.close(fig)
    return fig

def plot_SankeyDiagram(cluster_composition_dict,
                       labels_dict,color_map,
                       title='Sankey Diagram for cluster assignment vs true labels'):
    
    nodes = [label for label in labels_dict.values()]+[cluster_id for cluster_id in cluster_composition_dict.keys()]
    node_colors = list(color_map.values()) + ['white']*20
    
    links = {
        "source": [],  # starting node indices
        "dest"  : [],  # ending node indices
        "count" : [],  # magnitude of flow
        "color" : []   # color of the link
    }
    
    for cluster_id, labels in cluster_composition_dict.items():
        unique_labels, count = th.unique(labels, return_counts=True)
        links['source'].extend(unique_labels.tolist())
        links['dest'].extend([cluster_id + 10 ]* len(unique_labels))
        links['count'].extend(count.tolist())
        links['color'].extend([color_map[labels_dict[label.item()]] for label in unique_labels])
    
    fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,            # padding between nodes
        thickness=20,      # thickness of nodes
        line=dict(color="black", width=0.5),
        label=nodes,
        color=node_colors  # colors for each node
    ),
    link=dict(
        source=links["source"],
        target=links["dest"],
        value=links["count"],
        color=links["color"]
        )
    )])
    # Set the title and display the diagram
    fig.update_layout(
    width=800,height=500,
    font_color="black",
    font_size=18,
    font_weight="bold"
)
    return fig