# %%
import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt        
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import h5py



h5_obs = h5py.File("/p/tmp/bochow/Daily data/dailyData.h5",'r')

sea_ice_obs= np.array(h5_obs["Dataset1"])#[49*12:-1,:,:]
sea_ice_obs_copy = sea_ice_obs.copy() 
sea_ice_obs_copy[sea_ice_obs==0] =122
sea_ice_obs_copy[sea_ice_obs ==122] = 0


mask_train = np.ones(sea_ice_obs_copy.shape[0], dtype=bool)
mask_train[::9] = 0

print(sea_ice_obs_copy[mask_train,:,:].shape)
print(sea_ice_obs_copy[::9,:,:].shape)
print(sea_ice_obs_copy.shape)

sys.exit()

h5 = h5py.File("/p/tmp/bochow/climatereconstructionAI-1.0.0/FREVA-CLINT-climatereconstructionAI-2fc5e62/data/sea_ice_cover_3D_72x72.h5",'r')
sea_ice_full = np.array(h5["Dataset1"])#[49*12:-1,:,:]

mask = np.zeros((sea_ice_full[-1,:,:].shape))
mask[sea_ice_full[-3,:,:]==122]=1#

sea_ice_obs_copy[:,mask==1] =122


s=plt.imshow(sea_ice_obs_copy[-1,:,:])
plt.colorbar(s)
plt.savefig("/p/tmp/bochow/Daily data/plot.png")
plt.clf()
plt.imshow(mask)
plt.savefig("/p/tmp/bochow/Daily data/mask.png")


h = h5py.File('/p/tmp/bochow/climatereconstructionAI-1.0.0/FREVA-CLINT-climatereconstructionAI-2fc5e62/climate/val_large/val_daily.h5', 'w')
dset = h.create_dataset('sic', data=sea_ice_obs_copy[::9,:,:])
h.close()

h = h5py.File('/p/tmp/bochow/climatereconstructionAI-1.0.0/FREVA-CLINT-climatereconstructionAI-2fc5e62/climate/data_large/train_daily.h5', 'w')
dset = h.create_dataset('sic', data=sea_ice_obs_copy[mask_train,:,:])
h.close()

