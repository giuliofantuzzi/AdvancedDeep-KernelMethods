#--------------------------------------------------------------------------------
# Libraries and modules
#--------------------------------------------------------------------------------

# Standard libraries
import pickle
# Data preparation libraries
from utils import FashionMNIST_DataPrep
# Plotting libraries
from utils import plot_clusters_composition,plot_Spectrum_and_CVR,plot_SankeyDiagram
# Dimensionality reduction libraries
from sklearn.decomposition import KernelPCA
# Clustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score,homogeneity_completeness_v_measure

#--------------------------------------------------------------------------------
# Global variables
#--------------------------------------------------------------------------------

REDUCED_TRAININGSET_SIZE = 7500  # Number of samples to use for the (reduced) training
BEST_GAMMA = 0.05                # Gamma Hyperparameter for Gaussian KernelPCA
ZOOM_PC = 100                    # Number of principal components to plot in the kPCA spectrum

labels_dict = { 0: 'T-shirt/top',
                1: 'Trouser',
                2: 'Pullover',
                3: 'Dress',
                4: 'Coat',
                5: 'Sandal',
                6: 'Shirt',
                7: 'Sneaker',
                8: 'Bag',
                9: 'Ankle boot' }


color_map ={'T-shirt/top':'#636EFA',
             'Trouser':'#EF553B',
             'Pullover':'#00CC96',
             'Dress':'#AB63FA',
             'Coat':'#FFA15A',
             'Sandal':'#19D3F3',
             'Shirt':'#FF6692',
             'Sneaker':'#B6E880',
             'Bag':'#FF97FF',
             'Ankle boot':'#FECB52'}

#--------------------------------------------------------------------------------
# Data preparation
#--------------------------------------------------------------------------------

print('>> Performing Data Preparation')

_, _, _, _, X_train_reduced, y_train_reduced = FashionMNIST_DataPrep(reduced_trainingset_size=REDUCED_TRAININGSET_SIZE)

#--------------------------------------------------------------------------------
# Clustering
#--------------------------------------------------------------------------------

print('>> Performing Clustering')

# Dimensionality Reduction with Kernel PCA 
kPCA_train = X_train_reduced.view(-1, 28*28)
#kpca = KernelPCA(n_components=10,kernel='rbf', gamma=BEST_GAMMA, random_state=16)
kpca = KernelPCA(kernel='rbf', gamma=BEST_GAMMA, random_state=16)
projections = kpca.fit_transform(kPCA_train)

# Agglomerative Clustering
clusterer_AC = AgglomerativeClustering(n_clusters=10, linkage='ward')
clusterer_AC.fit(projections[:,:10]) # Note: as specified in the task, we only use the first 10 components for clustering
clusters_composition_dict_AC = {cluster_id : [] for cluster_id in range(10)}
for cluster_id in clusters_composition_dict_AC.keys():
    clusters_composition_dict_AC[cluster_id] = y_train_reduced[clusterer_AC.labels_ == cluster_id]

with open('models/clusterer_AC.pkl', 'wb') as file:
    pickle.dump(clusterer_AC, file)

#--------------------------------------------------------------------------------
# Plots
#--------------------------------------------------------------------------------
# Plotting clusters composition
fig_AC=plot_clusters_composition(clusters_composition_dict_AC,
                                 labels_dict=labels_dict,color_map=color_map,
                                 suptitle='Composition of clusters w.r.t. true labels')
fig_AC.savefig('plots/clustering/clusters_composition_AC.png',dpi=200)

# kPCA Spectrum (zoom on first principal components)
fig_spectrum = plot_Spectrum_and_CVR(kpca.eigenvalues_,n_components_plot=ZOOM_PC)
fig_spectrum.savefig('plots/clustering/kPCA_spectrum.png',dpi=200)
# Sankey's Diagram
fig_sankey = plot_SankeyDiagram(clusters_composition_dict_AC,labels_dict,color_map,None)
fig_sankey.write_image(f"plots/clustering/sankey_diagram.png", width=800, height=500,scale=3)

#--------------------------------------------------------------------------------
# Evaluation Metrics
#--------------------------------------------------------------------------------
# (1) Normalized mutual information: how much (normalized) information is shared between true labels and predicted clusters
nmi = normalized_mutual_info_score(labels_true=y_train_reduced,
                                   labels_pred=clusterer_AC.labels_)
# (2) Adjusted Rand score: similarity between the true labels and the predicted clusters, adjusting for chance
ars = adjusted_rand_score(labels_true=y_train_reduced,
                          labels_pred=clusterer_AC.labels_)
# (3) Homogeneity : whether each cluster contains only members of a single class
#     Completeness: whether all members of a class are assigned to the same cluster
#     V-Measure   : harmonic mean of the previous two
homogeneity, completeness,v_measure = homogeneity_completeness_v_measure(labels_true=y_train_reduced,
                                                                         labels_pred=clusterer_AC.labels_)

print('-'*60)
print('Clustering evaluation metrics')
print('-'*60)
print(f'* Normalized Mutual Information: {nmi}')
print(f'* Adjusted Rand score          : {ars}')
print(f'* Homogeneity                  : {homogeneity}')
print(f'* Completeness                 : {completeness}')
print(f'* V-Measure                    : {v_measure}')
print('-'*60)