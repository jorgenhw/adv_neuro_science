# import some functionality
from datetime import datetime

now = datetime.now()
print('Starting script:',now.strftime("%H:%M:%S"))
import os
import pip
import pickle

from nilearn.image import index_img, concat_imgs
import pandas as pd
import numpy as np
from nilearn.image import new_img_like, load_img
from nilearn.plotting import plot_stat_map, plot_img, show
from nilearn import decoding
from nilearn.decoding import SearchLight
from sklearn import naive_bayes, model_selection #import GaussianNB
from nilearn.plotting import plot_stat_map, plot_img, show
from nilearn import decoding
from sklearn import naive_bayes, model_selection #import GaussianNB
from nilearn.decoding import SearchLight
from sklearn.svm import LinearSVC
from nilearn.image import new_img_like, load_img
from nilearn.plotting import plot_stat_map, plot_img, show
from nilearn.image import new_img_like, load_img, index_img, clean_img
from sklearn.model_selection import train_test_split, GroupKFold

# Getting back the objects:

for i in range(21,22):
    text = "Loading file %d\n" % (i+1)
    print(text)
    file_name='/work/82777/WordFace_first_level_z_maps_all_trials_all_par_'+str(i)+'.pkl'
    f = open(file_name, 'rb')
    conditions_label, z_maps = pickle.load(f)
    n_trials=len(conditions_label)
    now = datetime.now()
    text = "Reshaping participant %d\n: %s " % (i+1, now.strftime("%H:%M:%S"))
    print(text)

    # Reshaping data------------------------------
    
    idx_neg=[int(ii) for ii in range(len(conditions_label)) if conditions_label[ii]=='image_neg']
    idx_pos=[int(ii) for ii in range(len(conditions_label)) if conditions_label[ii]=='image_pos']

    #Concatenate trials using nilearn
    idx=np.concatenate((idx_neg, idx_pos))

    conditions=np.array(conditions_label)[idx]
    z_maps_conc=concat_imgs(z_maps)
    print(z_maps_conc.shape)
    z_maps_img = index_img(z_maps_conc, idx)

    #Make an index for spliting fMRI data with same size as class labels
    idx2=np.arange(conditions.shape[0])

    # create training and testing vars on the basis of class labels
    idx1,idx2, conditions1,  conditions2 = train_test_split(idx2,conditions, test_size=0.2)
    #print(idx1, idx2)

    # Reshaping data------------------------------
    from nilearn.image import index_img
    fmri_img1 = index_img(z_maps_img, idx1)
    fmri_img2 = index_img(z_maps_img, idx2)
    #Check data sizes
    print(fmri_img1.shape)
    print(fmri_img2.shape)

    #Whole brain mask. Using one from a one participant for all
    mask_wb_filename='/work/82777/BIDS/derivatives/sub-0096/anat/sub-0096_acq-T1sequence_run-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'
    #Load the whole brain mask
    mask_img = load_img(mask_wb_filename)

    # .astype() makes a copy.
    process_mask = mask_img.get_fdata().astype('int32')
    #Set slices below x in the z-dimension to zero (in voxel space)
    process_mask[..., :30] = 0
    #Set slices above x in the z-dimension to zero (in voxel space)
    process_mask[..., 160:] = 0
    process_mask_img = new_img_like(mask_img, process_mask)

    #########################################################################
    # Run search light analysis
    now = datetime.now()
    text = "Starting searchlight participant %d\n: %s" % (i+1,now.strftime("%H:%M:%S"))
    print(text)
   
    # The radius is the one of the Searchlight sphere that will scan the volume
    searchlight = SearchLight(
        mask_img,
        estimator=LinearSVC(penalty='l2',dual='auto',max_iter=2000),
        process_mask_img=process_mask_img,
        radius=5, n_jobs=-1,
        verbose=10, cv=10)
    searchlight.fit(fmri_img1, conditions1)

    now = datetime.now()
    print('Finishing searchlight and saving:',now.strftime("%H:%M:%S"))
    
    import pickle
    file_name='/work/82777/WordFace_searchlight_'+str(i)+'.pkl'
    # Saving the objects:
    f = open(file_name, 'wb')
    pickle.dump(searchlight, f)
    f.close()

