#--------------------------------------------------------------------------------
# Libraries and modules
#--------------------------------------------------------------------------------
# Standard libraries
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
_, _, X_test, y_test = FashionMNIST_DataPrep()

with open('models/fully_supervised/best_SVC.pkl', 'rb') as file:
    best_SVC = pickle.load(file)
with open('models/fully_supervised/best_FCNN.pkl', 'rb') as file:
    best_FCNN = pickle.load(file)
with open('models/fully_supervised/best_CNN.pkl', 'rb') as file:
    best_CNN = pickle.load(file)
    
#--------------------------------------------------------------------------------
# Models evaluation
#--------------------------------------------------------------------------------

# Predictions

y_pred_SVC  = best_SVC.predict(X_test.view(-1, 28*28)).astype(int)
y_pred_FCNN = best_FCNN.predict(X_test.view(-1, 28*28))
y_pred_CNN  = best_CNN.predict(X_test)

# Convert numeric labels to string labels to have a better (visual) coutput
y_test_label = [labels_dict[label.item()] for label in y_test]
y_pred_SVC   = [labels_dict[label] for label in y_pred_SVC]
y_pred_FCNN  = [labels_dict[label] for label in y_pred_FCNN]
y_pred_CNN   = [labels_dict[label] for label in y_pred_CNN]


# Classification reports 
classification_report_SVC = classification_report(y_test_label, y_pred_SVC,zero_division=np.nan)
classification_report_FCNN = classification_report(y_test_label, y_pred_FCNN,zero_division=np.nan)
classification_report_CNN = classification_report(y_test_label, y_pred_CNN,zero_division=np.nan)

print('-'*80)
print('>> Classification report for the best SVC model:')
print(classification_report_SVC)
print('-'*80)
print('>> Classification report for the best FCNN model:')
print(classification_report_FCNN)
print('-'*80)
print('>> Classification report for the best CNN model:')
print(classification_report_CNN)
print('-'*80)

# Confusion Matrices
plot_CM__SVC = plot_ConfusionMatrix(y_true=y_test_label,y_pred=y_pred_SVC ,labels=[label for label in labels_dict.values()],cmap='Reds')
plot_CM_FCNN = plot_ConfusionMatrix(y_true=y_test_label,y_pred=y_pred_FCNN,labels=[label for label in labels_dict.values()],cmap='Reds')
plot_CM_CNN  = plot_ConfusionMatrix(y_true=y_test_label,y_pred=y_pred_CNN ,labels=[label for label in labels_dict.values()],cmap='Reds')


plot_CM__SVC.write_image(f"plots/evaluation/CM_FullySupervised_SVC.png", width=500, height=500,scale=3)
plot_CM_FCNN.write_image(f"plots/evaluation/CM_FullySupervised_FCNN.png", width=500, height=500,scale=3)
plot_CM_CNN.write_image(f"plots/evaluation/CM_FullySupervised_CNN.png", width=500, height=500,scale=3)


