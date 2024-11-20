import plotly.express as px
from sklearn.metrics import confusion_matrix


def plot_ConfusionMatrix(y_true,y_pred,labels,cmap="Blues"):
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig = px.imshow(cm,
                    labels=dict(x="Predicted", y="Actual"),
                    x=labels,y=labels,
                    text_auto=True,
                    color_continuous_scale=cmap
                    )
    fig.update_xaxes(side="top")
    fig.update_layout(coloraxis_showscale=False)
    # Adding borders around each cell
    for i in range(len(labels)):
        for j in range(len(labels)):
            fig.add_shape(
                type="rect",
                x0=j - 0.5, x1=j + 0.5,
                y0=i - 0.5, y1=i + 0.5,
                line=dict(color="black", width=1)
            )
    return fig