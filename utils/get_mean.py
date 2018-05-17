import numpy as np

print("Loading_data..")
PATH_TO_DATA = '/media/jomari/HDD/VIGNET/inpainting/train/npz/train.npz'
vignet = np.load(PATH_TO_DATA)
data = vignet['images']

R_mean = data[:,:,:,0].mean()
G_mean = data[:,:,:,1].mean()
B_mean = data[:,:,:,2].mean()

print(R_mean,G_mean,B_mean)
