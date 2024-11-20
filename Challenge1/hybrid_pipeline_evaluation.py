#--------------------------------------------------------------------------------
# Libraries and modules
#--------------------------------------------------------------------------------
# Standard libraries
import torch as th
import numpy as np
import pickle
# Plotting libraries
from utils import plot_ConfusionMatrix
# Data preparation libraries
from utils import FashionMNIST_DataPrep
# Model evaluation libraries
from sklearn.metrics import classification_report

#--------------------------------------------------------------------------------
# Global variables
#--------------------------------------------------------------------------------
REDUCED_TRAININGSET_SIZE=7500

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
#--------------------------------------------------------------------------------
# Data and Models import
#--------------------------------------------------------------------------------
# Note: evaluation of the models will be performed on the original test set
_, _, X_test, y_test, _, y_train_reduced = FashionMNIST_DataPrep(reduced_trainingset_size=REDUCED_TRAININGSET_SIZE)

with open('models/clusterer_AC.pkl', 'rb') as file:
    clusterer_AC = pickle.load(file)
with open('models/hybrid_pipeline/best_SVC.pkl', 'rb') as file:
    best_SVC = pickle.load(file)
with open('models/hybrid_pipeline/best_FCNN.pkl', 'rb') as file:
    best_FCNN = pickle.load(file)
with open('models/hybrid_pipeline/best_CNN.pkl', 'rb') as file:
    best_CNN = pickle.load(file)
    
#--------------------------------------------------------------------------------
# Models evaluation
#--------------------------------------------------------------------------------

# Mapping from cluster id to label
clusters_composition_dict = {cluster_id : [] for cluster_id in range(10)}

for cluster_id in clusters_composition_dict.keys():
    cluster_distribution = {label : 0 for label in range(10)}
    cluster_labels = y_train_reduced[clusterer_AC.labels_ == cluster_id]
    unique, counts = th.unique(cluster_labels, return_counts=True) 
    for label, count in zip(unique, counts):
        cluster_distribution[label.item()] = count.item()
    clusters_composition_dict[cluster_id] = cluster_distribution

def cluster_to_label_MajorityMapping(cluster_id,clusters_composition_dict):
    cluster_distribution = clusters_composition_dict[cluster_id]
    majority_label = max(cluster_distribution, key=cluster_distribution.get)
    return majority_label


def cluster_to_label_ProbabilisticMapping(cluster_id,clusters_composition_dict):
    cluster_distribution = clusters_composition_dict[cluster_id]
    labels = np.array([label for label in cluster_distribution.keys()])
    probabilities = np.array([prob for prob in cluster_distribution.values()]) / sum(cluster_distribution.values())
    return np.random.choice(labels,size=None,p=probabilities)


# Models evaluation

y_pred_cluster_SVC = best_SVC.predict(X_test.view(-1, 28*28)).astype(int)
y_pred_cluster_FCNN = best_FCNN.predict(X_test.view(-1, 28*28))
y_pred_cluster_CNN = best_CNN.predict(X_test)

# Convert numeric labels to string labels to have a better (visual) coutput
y_test_label = [labels_dict[label.item()] for label in y_test]

y_pred_label_MajorityMapping_SVC  = [labels_dict[cluster_to_label_MajorityMapping(pred_cluster,clusters_composition_dict)] for pred_cluster in y_pred_cluster_SVC]
y_pred_label_MajorityMapping_FCNN = [labels_dict[cluster_to_label_MajorityMapping(pred_cluster,clusters_composition_dict)] for pred_cluster in y_pred_cluster_FCNN]
y_pred_label_MajorityMapping_CNN  = [labels_dict[cluster_to_label_MajorityMapping(pred_cluster,clusters_composition_dict)] for pred_cluster in y_pred_cluster_CNN]

y_pred_label_ProbabilisticMapping_SVC  = [labels_dict[cluster_to_label_ProbabilisticMapping(pred_cluster,clusters_composition_dict)] for pred_cluster in y_pred_cluster_SVC]
y_pred_label_ProbabilisticMapping_FCNN = [labels_dict[cluster_to_label_ProbabilisticMapping(pred_cluster,clusters_composition_dict)] for pred_cluster in y_pred_cluster_FCNN]
y_pred_label_ProbabilisticMapping_CNN  = [labels_dict[cluster_to_label_ProbabilisticMapping(pred_cluster,clusters_composition_dict)] for pred_cluster in y_pred_cluster_CNN]


# Classification reports 
classification_report_MajorityMapping_SVC = classification_report(y_test_label, y_pred_label_MajorityMapping_SVC,zero_division=np.nan)
classification_report_MajorityMapping_FCNN = classification_report(y_test_label, y_pred_label_MajorityMapping_FCNN,zero_division=np.nan)
classification_report_MajorityMapping_CNN = classification_report(y_test_label, y_pred_label_MajorityMapping_CNN,zero_division=np.nan)

classification_report_ProbabilisticMapping_SVC = classification_report(y_test_label, y_pred_label_ProbabilisticMapping_SVC,zero_division=np.nan)
classification_report_ProbabilisticMapping_FCNN = classification_report(y_test_label, y_pred_label_ProbabilisticMapping_FCNN,zero_division=np.nan)
classification_report_ProbabilisticMapping_CNN = classification_report(y_test_label, y_pred_label_ProbabilisticMapping_CNN,zero_division=np.nan)


print('-'*80)
print('>> Classification report for the best SVC model [Majority Mapping]:')
print(classification_report_MajorityMapping_SVC)
print('-'*80)
print('>> Classification report for the best FCNN model [Majority Mapping]:')
print(classification_report_MajorityMapping_FCNN)
print('-'*80)
print('>> Classification report for the best CNN model [Majority Mapping]:')
print(classification_report_MajorityMapping_CNN)
print('-'*80)
print('>> Classification report for the best SVC model [Probabilistic Mapping]:')
print(classification_report_ProbabilisticMapping_SVC)
print('-'*80)
print('>> Classification report for the best FCNN model [Probabilistic Mapping]:')
print(classification_report_ProbabilisticMapping_FCNN)
print('-'*80)
print('>> Classification report for the best CNN model [Probabilistic Mapping]:')
print(classification_report_ProbabilisticMapping_CNN)
print('-'*80)

# Confusion Matrices
plot_CM_MajorityMapping_SVC=plot_ConfusionMatrix(y_true=y_test_label,y_pred=y_pred_label_MajorityMapping_SVC,labels=[label for label in labels_dict.values()],cmap='Blues')
plot_CM_ProbabilisticMapping_SVC=plot_ConfusionMatrix(y_true=y_test_label,y_pred=y_pred_label_ProbabilisticMapping_SVC,labels=[label for label in labels_dict.values()],cmap='Greens')
plot_CM_MajorityMapping_FCNN=plot_ConfusionMatrix(y_true=y_test_label,y_pred=y_pred_label_MajorityMapping_FCNN,labels=[label for label in labels_dict.values()],cmap='Blues')
plot_CM_ProbabilisticMapping_FCNN=plot_ConfusionMatrix(y_true=y_test_label,y_pred=y_pred_label_ProbabilisticMapping_FCNN,labels=[label for label in labels_dict.values()],cmap='Greens')
plot_CM_MajorityMapping_CNN=plot_ConfusionMatrix(y_true=y_test_label,y_pred=y_pred_label_MajorityMapping_CNN,labels=[label for label in labels_dict.values()],cmap='Blues')
plot_CM_ProbabilisticMapping_CNN=plot_ConfusionMatrix(y_true=y_test_label,y_pred=y_pred_label_ProbabilisticMapping_CNN,labels=[label for label in labels_dict.values()],cmap='Greens')

plot_CM_MajorityMapping_SVC.write_image(f"plots/evaluation/CM_HybridPipeline_MajorityMapping_SVC.png", width=500, height=500,scale=3)
plot_CM_ProbabilisticMapping_SVC.write_image(f"plots/evaluation/CM_HybridPipeline_ProbabilisticMapping_SVC.png", width=500, height=500,scale=3)
plot_CM_MajorityMapping_FCNN.write_image(f"plots/evaluation/CM_HybridPipeline_MajorityMapping_FCNN.png", width=500, height=500,scale=3)
plot_CM_ProbabilisticMapping_FCNN.write_image(f"plots/evaluation/CM_HybridPipeline_ProbabilisticMapping_FCNN.png", width=500, height=500,scale=3)
plot_CM_MajorityMapping_CNN.write_image(f"plots/evaluation/CM_HybridPipeline_MajorityMapping_CNN.png", width=500, height=500,scale=3)
plot_CM_ProbabilisticMapping_CNN.write_image(f"plots/evaluation/CM_HybridPipeline_ProbabilisticMapping_CNN.png", width=500, height=500,scale=3)
