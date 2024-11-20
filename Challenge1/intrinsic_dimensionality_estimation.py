#--------------------------------------------------------------------------------
# Libraries and modules
#--------------------------------------------------------------------------------
# Data preparation libraries
from utils import FashionMNIST_DataPrep
# DADApy for intrinsic dimensionality estimation
from dadapy import data

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
# Data import
#--------------------------------------------------------------------------------
# (Reduced) Training set with original images
_, _, _, _, X_train_reduced, y_train_reduced = FashionMNIST_DataPrep(reduced_trainingset_size=REDUCED_TRAININGSET_SIZE)

#--------------------------------------------------------------------------------
# Intrinsic dimensionality estimation
#--------------------------------------------------------------------------------
_data = data.Data(X_train_reduced.reshape(-1,28*28).numpy())
_data.remove_identical_points()
id_twoNN, _, r = _data.compute_id_2NN()
print(f"Intrinsic dimensionality of the Fashion-MNIST data: {id_twoNN}")