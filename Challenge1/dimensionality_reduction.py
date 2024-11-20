#--------------------------------------------------------------------------------
# Libraries and modules
#--------------------------------------------------------------------------------

# Standard libraries
from tqdm import tqdm
# Data preparation libraries
from utils import FashionMNIST_DataPrep
# Plotting libraries
from utils import plot_2PC,plot_3PC
# Dimensionality reduction libraries
from sklearn.decomposition import PCA,KernelPCA

#--------------------------------------------------------------------------------
# Global variables
#--------------------------------------------------------------------------------

REDUCED_TRAININGSET_SIZE = 7500

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
# Standard PCA
#--------------------------------------------------------------------------------

print('>> Performing Standard PCA')

PCA_train = X_train_reduced.view(-1, 28*28)
pca = PCA(random_state=16)
projections=pca.fit_transform(PCA_train)

plot2d = plot_2PC(projections,y_train_reduced,labels_dict=labels_dict,color_map=color_map)
plot3d=plot_3PC(projections,y_train_reduced,labels_dict=labels_dict,color_map=color_map)
plot2d.write_image(f"plots/PCA/PCA_2PC.png", width=800, height=500,scale=3)
plot3d.write_html(f"plots/PCA/PCA_3PC.html")

#--------------------------------------------------------------------------------
# Radial Basis Function (Gaussian) PCA
#--------------------------------------------------------------------------------

kPCA_train = X_train_reduced.view(-1, 28*28)

gammas = [0.001,0.005,0.01,0.05,0.1,0.5,1,5]

for gamma in tqdm(gammas,desc=f'>> Gaussian Kernel PCA'):
    # 1) Perform kPCA
    kpca = KernelPCA(n_components=3, kernel='rbf', gamma=gamma, random_state=16)
    projections = kpca.fit_transform(kPCA_train)
    # 2) Plots of principal components
    plot2d = plot_2PC(projections,y_train_reduced,
                      labels_dict=labels_dict,color_map=color_map,
                      title=f'Fashion MNIST (2 principal components) - gamma={gamma}')
    plot3d = plot_3PC(projections,y_train_reduced,
                      labels_dict=labels_dict,color_map=color_map,
                      title=f'Fashion MNIST (3 principal components) - gamma={gamma}')
    # 3) Save plots and append them to list
    plot2d.write_image(f"plots/rbfPCA/rbfPCA_2PC_{gamma}.png", width=800, height=500,scale=3)
    plot3d.write_html(f"plots/rbfPCA/rbfPCA_3PC_{gamma}.html")

#--------------------------------------------------------------------------------
# Polynomial PCA
#--------------------------------------------------------------------------------

degrees = [2,3,4,5]

for deg in tqdm(degrees,desc=f'>> Polynomial Kernel PCA'):
    # 1) Perform kPCA
    kpca = KernelPCA(n_components=3,kernel='poly', degree=deg, random_state=16)
    projections = kpca.fit_transform(kPCA_train)
    # 2) Plots of principal components
    plot2d = plot_2PC(projections,y_train_reduced,
                      labels_dict=labels_dict,color_map=color_map,
                      title=f'Fashion MNIST (2 principal components) - degree={deg}')
    plot3d = plot_3PC(projections,y_train_reduced,
                      labels_dict=labels_dict,color_map=color_map,
                      title=f'Fashion MNIST (3 principal components) - degree={deg}')
    # 3) Save plots and append them to list
    plot2d.write_image(f"plots/polyPCA/polyPCA_2PC_{deg}.png", width=800, height=500,scale=3)
    plot3d.write_html(f"plots/polyPCA/polyPCA_3PC_{deg}.html")
