# %%
import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt        
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import h5py


# %%
h5 = h5py.File("/p/tmp/bochow/climatereconstructionAI-1.0.0/FREVA-CLINT-climatereconstructionAI-2fc5e62/data/sea_ice_cover_3D_72x72.h5",'r')

sea_ice_full = np.array(h5["Dataset1"])#[49*12:-1,:,:]
print(sea_ice_full.shape)
h = h5py.File('/p/tmp/bochow/climatereconstructionAI-1.0.0/FREVA-CLINT-climatereconstructionAI-2fc5e62/climate/test_large/sea_ice_cover_3D_72x72_sic_full.h5', 'w')
dset = h.create_dataset('sic', data=sea_ice_full[:,:,:])
h.close()



mask = np.zeros((sea_ice_full[-1,:,:].shape))
mask[sea_ice_full[-3,:,:]==122]=1

#plt.imshow(mask)
#plt.savefig("/p/tmp/bochow/climatereconstructionAI-1.0.0/FREVA-CLINT-climatereconstructionAI-2fc5e62/mask_72.png")
#np.save("/p/tmp/bochow/climatereconstructionAI-1.0.0/FREVA-CLINT-climatereconstructionAI-2fc5e62/data/continent_mask_72.npy", mask)

sea_ice_full = sea_ice_full#[49*12:-1,:,:]
    
h5_era = h5py.File("/p/tmp/bochow/climatereconstructionAI-1.0.0/FREVA-CLINT-climatereconstructionAI-2fc5e62/data/era5_2.5_1950-2021.h5",'r')
era5 = np.array(h5_era["Dataset1"])[0:-12*9,:,:]

siconc_h5 = h5py.File("/p/tmp/bochow/climatereconstructionAI-1.0.0/FREVA-CLINT-climatereconstructionAI-2fc5e62/data/SIconc_72x72_1979-2017.h5",'r')
siconc = np.array(siconc_h5["Dataset1"])[:,:,:]

#mask = np.load("/p/tmp/bochow/climatereconstructionAI-1.0.0/FREVA-CLINT-climatereconstructionAI-2fc5e62/data/continent_mask.npy")




# %%
mask_missing_data = np.ones((sea_ice_full.shape))
print(mask_missing_data.shape)
mask_missing_data[[(mask!=1) & (sea_ice_full ==122)]] = 0


print(mask_missing_data.shape)





# %%
h = h5py.File('/p/tmp/bochow/climatereconstructionAI-1.0.0/FREVA-CLINT-climatereconstructionAI-2fc5e62/masks/sea_ice_missmask_full.h5', 'w')
dset = h.create_dataset('sic', data=mask_missing_data)
h.close()

"""
# %%
filename = '/p/tmp/bochow/climatereconstructionAI-1.0.0/FREVA-CLINT-climatereconstructionAI-2fc5e62/masks/sea_ice_missmask.h5'

with h5py.File(filename, "r") as f:
    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]

    # Get the data
    data = list(f[a_group_key])
    
    
h5 = h5py.File(filename,'r')
sic = h5["sic"]
print(sic)



# %%
siconc[siconc==120] =122

# %%
siconc.shape

# %%
h = h5py.File('/p/tmp/bochow/climatereconstructionAI-1.0.0/FREVA-CLINT-climatereconstructionAI-2fc5e62/climate/data_large/sea_conc_obs_train.h5', 'w')
dset = h.create_dataset('sic', data=siconc[0:-2,:,:])
h.close()

h = h5py.File('/p/tmp/bochow/climatereconstructionAI-1.0.0/FREVA-CLINT-climatereconstructionAI-2fc5e62/climate/val_large/sea_conc_obs_val.h5', 'w')
dset = h.create_dataset('sic', data=siconc[-1,:,:])
h.close()

"""
