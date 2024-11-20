#--------------------------------------------------------------------------------
# Libraries and modules
#--------------------------------------------------------------------------------
import math
import pickle
from architectures import FCNN,CNN
# Suppress InconsistentVersionWarning
# (used locally (MacOS) just to suppress warnings deriving from cuda, nothing to worry about)
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

#--------------------------------------------------------------------------------
# Global variables
#--------------------------------------------------------------------------------
N = 5
MODEL_TYPE = "fully_supervised/" # or "hybrid_pipeline/"  

#--------------------------------------------------------------------------------
# CROSS-VALIDATION REPORTS
#--------------------------------------------------------------------------------

# (1) SVC
model_path = 'tuning/' + MODEL_TYPE +'grid_search_SVC.pkl'
with open(model_path,'rb') as file:
    grid_search_SVC = pickle.load(file)
    
mean_test_score_SVC = grid_search_SVC.cv_results_['mean_test_score'][grid_search_SVC.best_index_]
std_test_score_SVC = grid_search_SVC.cv_results_['std_test_score'][grid_search_SVC.best_index_]
mean_fit_time_SVC = grid_search_SVC.cv_results_['mean_fit_time'][grid_search_SVC.best_index_]
std_fit_time_SVC = grid_search_SVC.cv_results_['std_fit_time'][grid_search_SVC.best_index_]
mean_score_time_SVC = grid_search_SVC.cv_results_['mean_score_time'][grid_search_SVC.best_index_]
std_score_time_SVC = grid_search_SVC.cv_results_['std_score_time'][grid_search_SVC.best_index_]

print('-'*80)
print(f"SVC best results")
print(f"Best configuration  -> {grid_search_SVC.best_params_}")
print(f"Balanced Accuracy   -> {mean_test_score_SVC:.4f} +- {1.96*std_test_score_SVC/math.sqrt(N):.4f}")
print(f"Training time (s)   -> {mean_fit_time_SVC:.4f} +- {1.96*std_fit_time_SVC/math.sqrt(N):.4f}")
print(f"Prediction time (s) -> {mean_score_time_SVC:.4f} +- {1.96*std_score_time_SVC/math.sqrt(N):.4f}")
print('-'*80)

# (2) FCNN
model_path = 'tuning/' + MODEL_TYPE +'grid_search_FCNN.pkl'
with open(model_path,'rb') as file:
    grid_search_FCNN = pickle.load(file)
    
mean_test_score_FCNN = grid_search_FCNN.cv_results_['mean_test_score'][grid_search_FCNN.best_index_]
std_test_score_FCNN = grid_search_FCNN.cv_results_['std_test_score'][grid_search_FCNN.best_index_]
mean_fit_time_FCNN = grid_search_FCNN.cv_results_['mean_fit_time'][grid_search_FCNN.best_index_]
std_fit_time_FCNN = grid_search_FCNN.cv_results_['std_fit_time'][grid_search_FCNN.best_index_]
mean_score_time_FCNN = grid_search_FCNN.cv_results_['mean_score_time'][grid_search_FCNN.best_index_]
std_score_time_FCNN = grid_search_FCNN.cv_results_['std_score_time'][grid_search_FCNN.best_index_]

print(f"FCNN best results")
print(f"Best configuration  -> {grid_search_FCNN.best_params_}")
print(f"Balanced Accuracy   -> {mean_test_score_FCNN:.4f} +- {1.96*std_test_score_FCNN/math.sqrt(N):.4f}")
print(f"Training time (s)   -> {mean_fit_time_FCNN:.4f} +- {1.96*std_fit_time_FCNN/math.sqrt(N):.4f}")
print(f"Prediction time (s) -> {mean_score_time_FCNN:.4f} +- {1.96*std_score_time_FCNN/math.sqrt(N):.4f}")
print('-'*80)

# (3) CNN
model_path = 'tuning/' + MODEL_TYPE +'grid_search_CNN.pkl'
with open(model_path,'rb') as file:
    grid_search_CNN = pickle.load(file)
    
mean_test_score_CNN = grid_search_CNN.cv_results_['mean_test_score'][grid_search_CNN.best_index_]
std_test_score_CNN = grid_search_CNN.cv_results_['std_test_score'][grid_search_CNN.best_index_]
mean_fit_time_CNN = grid_search_CNN.cv_results_['mean_fit_time'][grid_search_CNN.best_index_]
std_fit_time_CNN = grid_search_CNN.cv_results_['std_fit_time'][grid_search_CNN.best_index_]
mean_score_time_CNN = grid_search_CNN.cv_results_['mean_score_time'][grid_search_CNN.best_index_]
std_score_time_CNN = grid_search_CNN.cv_results_['std_score_time'][grid_search_CNN.best_index_]

print(f"CNN best results")
print(f"Best configuration  -> {grid_search_CNN.best_params_}")
print(f"Balanced Accuracy   -> {mean_test_score_CNN:.4f} +- {1.96*std_test_score_CNN/math.sqrt(N):.4f}")
print(f"Training time (s)   -> {mean_fit_time_CNN:.4f} +- {1.96*std_fit_time_CNN/math.sqrt(N):.4f}")
print(f"Prediction time (s) -> {mean_score_time_CNN:.4f} +- {1.96*std_score_time_CNN/math.sqrt(N):.4f}")
print('-'*80)
#--------------------------------------------------------------------------------