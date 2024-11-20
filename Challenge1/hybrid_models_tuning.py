#--------------------------------------------------------------------------------
# Libraries and modules
#--------------------------------------------------------------------------------
# Standard libraries
import torch as th
import torch.nn as nn
import pickle
import math
# Data preparation libraries
from utils import FashionMNIST_DataPrep
# Models
from sklearn.svm import SVC
from architectures import FCNN,CNN
# Hyperparameter tuning libraries
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV

#--------------------------------------------------------------------------------
# Global variables
#--------------------------------------------------------------------------------

REDUCED_TRAININGSET_SIZE = 7500
TUNING_FOLDS = 5

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
# Data preparation
#--------------------------------------------------------------------------------

print('>> Performing Data Preparation')

# (Reduced) Training set with original images
_, _, _, _, X_train_reduced, _ = FashionMNIST_DataPrep(reduced_trainingset_size=REDUCED_TRAININGSET_SIZE)
# Labels from clustering procedure
with open('models/clusterer_AC.pkl', 'rb') as file:
    clustering_model = pickle.load(file)
y_train_cluster = th.tensor(clustering_model.labels_,dtype=th.long)



#--------------------------------------------------------------------------------
# Support Vector Classification (SVC)
#--------------------------------------------------------------------------------
# Given the non-linear geomety of data (see section 1), I'll go directly to a kernel SVM
# For Hyperparameter tuning, I'll use GridSearchCV
print('>> Performing Hyperparameter tuning for SVC')

# Hyperparameters grid
param_grid_SVC = [
    # Polynomial Kernel Grid
    {'kernel': ['poly'], 
     'C': [0.1, 1, 10,50],
     'degree': [2,3,4]
     }, 
    # Gaussian Kernel Grid
    {'kernel': ['rbf'],
     'C': [0.1, 1, 10,50],
     'gamma': [0.005,0.01,0.05,0.1,0.5,1,5]},
]

# Perform GridSearchCV and store the results with pickle
grid_search_SVC = GridSearchCV(SVC(), param_grid_SVC, cv=TUNING_FOLDS,scoring='balanced_accuracy',verbose=2)
grid_search_SVC.fit(X_train_reduced.view(-1, 28*28), y_train_cluster)
with open('tuning/hybrid_pipeline/grid_search_SVC.pkl', 'wb') as file:
    pickle.dump(grid_search_SVC, file)
with open('models/hybrid_pipeline/best_SVC.pkl', 'wb') as file:
    pickle.dump(grid_search_SVC.best_estimator_, file)
    
mean_test_score_SVC = grid_search_SVC.cv_results_['mean_test_score'][grid_search_SVC.best_index_]
std_test_score_SVC = grid_search_SVC.cv_results_['std_test_score'][grid_search_SVC.best_index_]
mean_fit_time_SVC = grid_search_SVC.cv_results_['mean_fit_time'][grid_search_SVC.best_index_]
std_fit_time_SVC = grid_search_SVC.cv_results_['std_fit_time'][grid_search_SVC.best_index_]
mean_score_time_SVC = grid_search_SVC.cv_results_['mean_score_time'][grid_search_SVC.best_index_]
std_score_time_SVC = grid_search_SVC.cv_results_['std_score_time'][grid_search_SVC.best_index_]

print('-'*80)
print(f"SVC best results")
print(f"Best configuration  -> {grid_search_SVC.best_params_}")
print(f"Balanced Accuracy   -> {mean_test_score_SVC:.4f} +- {1.96*std_test_score_SVC/math.sqrt(TUNING_FOLDS):.4f}")
print(f"Training time (s)   -> {mean_fit_time_SVC:.4f} +- {1.96*std_fit_time_SVC/math.sqrt(TUNING_FOLDS):.4f}")
print(f"Prediction time (s) -> {mean_score_time_SVC:.4f} +- {1.96*std_score_time_SVC/math.sqrt(TUNING_FOLDS):.4f}")

#--------------------------------------------------------------------------------
# Fully Connected Neural Network (FCNN)
#--------------------------------------------------------------------------------
print('>> Performing Hyperparameter tuning for FCNN')

# Initialize the NeuralNetClassifier
net = NeuralNetClassifier(
    FCNN,
    max_epochs=20,
    verbose=0,
    train_split=None,
    device= 'cuda' if th.cuda.is_available() else 'mps' if th.backends.mps.is_available() else 'cpu',
    module__input_dim=X_train_reduced.view(-1, 28*28).shape[1],  
    module__output_dim=th.unique(y_train_cluster).shape[0],           
    criterion=nn.CrossEntropyLoss,
    optimizer=th.optim.Adam
)

# Hyperparameters grid
param_grid_FCNN = {
    'module__hidden_dim': [128,256, 512],
    'module__non_linearity': [nn.ReLU(), nn.Sigmoid(),nn.Tanh()], 
    'module__dropout': [0.0,0.5],
    'lr': [1e-3,1e-2],
    'batch_size': [64,128,256],
}
# Perform GridSearchCV and store the results with pickle
grid_search_FCNN = GridSearchCV(net, param_grid_FCNN, refit=True, cv=TUNING_FOLDS,scoring='balanced_accuracy', verbose=2)
grid_search_FCNN.fit(X_train_reduced.view(-1, 28*28), y_train_cluster)
with open('tuning/hybrid_pipeline/grid_search_FCNN.pkl', 'wb') as file:
    pickle.dump(grid_search_FCNN, file)
with open('models/hybrid_pipeline/best_FCNN.pkl', 'wb') as file:
    pickle.dump(grid_search_FCNN.best_estimator_, file)

mean_test_score_FCNN = grid_search_FCNN.cv_results_['mean_test_score'][grid_search_FCNN.best_index_]
std_test_score_FCNN = grid_search_FCNN.cv_results_['std_test_score'][grid_search_FCNN.best_index_]
mean_fit_time_FCNN = grid_search_FCNN.cv_results_['mean_fit_time'][grid_search_FCNN.best_index_]
std_fit_time_FCNN = grid_search_FCNN.cv_results_['std_fit_time'][grid_search_FCNN.best_index_]
mean_score_time_FCNN = grid_search_FCNN.cv_results_['mean_score_time'][grid_search_FCNN.best_index_]
std_score_time_FCNN = grid_search_FCNN.cv_results_['std_score_time'][grid_search_FCNN.best_index_]

print(f"FCNN best results")
print(f"Best configuration  -> {grid_search_FCNN.best_params_}")
print(f"Balanced Accuracy   -> {mean_test_score_FCNN:.4f} +- {1.96*std_test_score_FCNN/math.sqrt(TUNING_FOLDS):.4f}")
print(f"Training time (s)   -> {mean_fit_time_FCNN:.4f} +- {1.96*std_fit_time_FCNN/math.sqrt(TUNING_FOLDS):.4f}")
print(f"Prediction time (s) -> {mean_score_time_FCNN:.4f} +- {1.96*std_score_time_FCNN/math.sqrt(TUNING_FOLDS):.4f}")

#--------------------------------------------------------------------------------
# Convolutional Neural Network (CNN)
#--------------------------------------------------------------------------------
print('>> Performing Hyperparameter tuning for CNN')        
        
# Initialize the NeuralNetClassifier
net = NeuralNetClassifier(
    CNN,
    max_epochs=20,
    verbose=0,
    train_split=None,
    device= 'cuda' if th.cuda.is_available() else 'mps' if th.backends.mps.is_available() else 'cpu',
    module__input_dim=X_train_reduced.shape[1],  
    module__output_dim=th.unique(y_train_cluster).shape[0],           
    criterion=nn.CrossEntropyLoss,
    optimizer=th.optim.Adam
)

# Hyperparameters grid
param_grid_CNN = {
    'module__non_linearity': [nn.ReLU(), nn.Sigmoid(),nn.Tanh()], 
    'module__pooling': [nn.MaxPool2d(kernel_size=2, stride=2),nn.AvgPool2d(kernel_size=2, stride=2)],
    'module__batchnorm': [True,False],
    'lr': [1e-3,1e-2],
    'batch_size': [64,128,256],
}

# Perform GridSearchCV and store the results with pickle
grid_search_CNN = GridSearchCV(net, param_grid_CNN, refit=True, cv=TUNING_FOLDS,scoring='balanced_accuracy', verbose=2)
grid_search_CNN.fit(X_train_reduced, y_train_cluster)
with open('tuning/hybrid_pipeline/grid_search_CNN.pkl', 'wb') as file:
    pickle.dump(grid_search_CNN, file)
with open('models/hybrid_pipeline/best_CNN.pkl', 'wb') as file:
    pickle.dump(grid_search_CNN.best_estimator_, file)

mean_test_score_CNN = grid_search_CNN.cv_results_['mean_test_score'][grid_search_CNN.best_index_]
std_test_score_CNN = grid_search_CNN.cv_results_['std_test_score'][grid_search_CNN.best_index_]
mean_fit_time_CNN = grid_search_CNN.cv_results_['mean_fit_time'][grid_search_CNN.best_index_]
std_fit_time_CNN = grid_search_CNN.cv_results_['std_fit_time'][grid_search_CNN.best_index_]
mean_score_time_CNN = grid_search_CNN.cv_results_['mean_score_time'][grid_search_CNN.best_index_]
std_score_time_CNN = grid_search_CNN.cv_results_['std_score_time'][grid_search_CNN.best_index_]    
print(f"Best configuration  -> {grid_search_CNN.best_params_}")
print(f"Balanced Accuracy   -> {mean_test_score_CNN:.4f} +- {1.96*std_test_score_CNN/math.sqrt(TUNING_FOLDS):.4f}")
print(f"Training time (s)   -> {mean_fit_time_CNN:.4f} +- {1.96*std_fit_time_CNN/math.sqrt(TUNING_FOLDS):.4f}")
print(f"Prediction time (s) -> {mean_score_time_CNN:.4f} +- {1.96*std_score_time_CNN/math.sqrt(TUNING_FOLDS):.4f}")

        