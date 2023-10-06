#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 14:31:08 2022

@author: lau
"""

#%% IMPORTS OF PACKAGES

import mne
from os.path import join
import matplotlib.pyplot as plt
plt.ion() #toggle interactive plotting
# import numpy as np

#%% PATHS FOR EXAMPLE ANALYSIS

meg_path = '/home/lau/analyses/undervisning_cs/raw/' + \
     '0108/20230928_000000/MEG/001.self_block1/files'
bem_path = '/home/lau/analyses/undervisning_cs/scratch/' + \ # bem = this is where all the MR data is. They are processed so that they fit with the MEG data
     'freesurfer/0108/bem' 
subjects_dir = '/home/lau/analyses/undervisning_cs/scratch/freesurfer/'      
raw_name = 'self_block1.fif'
fwd_name = 'self_block1-oct-6-src-5120-fwd.fif'

#%% READ RAW AND PLOT

raw = mne.io.read_raw_fif(join(meg_path, raw_name), preload=True)
raw.plot() ## what happens after 10 seconds? 
raw.compute_psd(n_jobs=-1).plot()
raw.compute_psd(n_jobs=-1, tmax=9).plot() 

#%% FILTER RAW

raw.filter(l_freq=None, h_freq=40, n_jobs=4) # alters raw in-place. We are filtering out the high frequencies (above 40 Hz).
raw.compute_psd(n_jobs=-1).plot() # plotting the power spectral density
raw.plot() # the thing we can see here is limited: Only great artificats are visible
#%% CONCATENATION OF RAWS - cannot be done without MaxFilter
# and setting the destination argument: We can concatenate the raws, but we cannot concatenate the events. We can't concatenete the raws, since this would assume that the head position is the same throughout all six trials.

# mne.preprocess.maxwell_filter

#%% FIND EVENTS

# events = mne.find_events(raw)#, min_duration=0.002) ## returns a numpy array
events = mne.find_events(raw, min_duration=0.002) ## returns a numpy array. Find the triggers in the rawfile in the MEG data.

mne.viz.plot_events(events) ## samples on x-axis
mne.viz.plot_events(events, sfreq=raw.info['sfreq']) ## 

# Now we have all the raw data that we recorded simultanously.

#%% SEGMENT DATA INTO EPOCHS

event_id = dict(self_positive=11, self_negative=12, button_press=23,
                incorrect_response=202) # coding the events into the event_id dictionary.
# reject = dict(mag=4e-12, grad=4000e-13, eog=250e-6) # T, T/m, V # if we think some of the data is bad, we can reject it. These are precoded values for exxluding data.
reject = None # for now, no rejections
epochs = mne.Epochs(raw, events, event_id, tmin=-0.200, tmax=1.000, # cutting data into 1.2 segments, tmin and tmax are the time before and after the event
                    baseline=(None, 0), reject=reject, preload=True, # baseline corrections means that we are subtracting the mean of the baseline from the data. This is done to remove the DC offset.
                    proj=False) # have proj=true if you want to reject!!

epochs.pick_types(meg=True, eog=False, ias=False, emg=False, misc=False, 
                  stim=False, syst=False)

epochs.plot()

#%% EVOKED - AVERAGE - projs not applied

evokeds = list()
for event in event_id: # for each event in the event_id dictionary

    evoked = epochs[event].average() # average across all epochs for each event
    evokeds.append(evoked)
    evoked.plot(window_title=evoked.comment)

# at around 100 ms we're seeing a reponse in the visual processing area.
#%% PROJS
mne.viz.plot_projs_topomap(evoked.info['projs'], evoked.info) # projectors: when loading data, stuff that are always in the sensor room such as electric fields, things in the hospital e.g. the elevator. 
                                                              # remember to do sanity checks on the data before applying the projectors - and report these sanity checks in the paper.
                                                              # these projectins are from an empty room recording so we can filter them out. This is an extra recording that we have.

epochs.apply_proj() # applying the projectors to the data

evokeds = list()
for event in event_id:

    evoked = epochs[event].average()
    evokeds.append(evoked)
    evoked.plot(window_title=evoked.comment)


#%% ARRAY OF INTEREST FOR CLASSIFICATIION

X = epochs.get_data() # X = the data that we want to classify on. That's the data in our epochs. Not the evokes, which are averages. X is all of the MEG data, that we'd later split into test/train. y would be our labels. 
                      # X.shape = 158 (trials), 306 (sensors), 1201 (time points)
                      # for classification, we need to flatten the data into a 2D array. 

#%% SOURCE RECONSTRUCTION

#%% read forward solution
fwd = mne.read_forward_solution(join(bem_path, fwd_name)) # forward solution has four components: 
                                                          # 1) source space, 
                                                          # 2) sensor space, 
                                                          # 3) transformation between source and sensor space, 
                                                          # 4) how the electric fields spread from the sources inside the head.
src = fwd['src'] # where are the sources
trans = fwd['mri_head_t'] # what's the transformation between mri and head: how do we get from the MRI to the head
info = epochs.info # where are the sensors? Coordinates of the sensors
bem_sol = fwd['sol'] # how do electric fields spread from the sources inside the head towards the sensors? 
bem = join(bem_path, '0108-5120-bem.fif') 

# inverse modelling: We are going from observing the magnetic fields to estimating the sources.
# Forward model = we are measuring


#%% plot source space
src.plot(trans=trans, subjects_dir=subjects_dir) # plotting the sensors source space. We can see that the sources are in the cortex.
src.plot(trans=fwd['mri_head_t'], subjects_dir=subjects_dir, head=True,
         skull='inner_skull')
mne.viz.plot_alignment(info, trans=trans, subject='0108', # plotting the allignment of the sensors and the sources
                       subjects_dir=subjects_dir, src=src,
                       bem=bem, dig=True, mri_fiducials=True)

#%% estimate covariance in the baseline to whiten magnetometers and 
#   gradiometers
noise_cov = mne.compute_covariance(epochs, tmax=0.000) # the noise perioed -200 ms until stimuli onset. 
noise_cov.plot(epochs.info) # not full range due to projectors projected out.
                            # The two plots shows the covariance of the magnetometers and the gradiometers.
                            # The magnetometers are more sensitive to the noise than the gradiometers.
                            # The other plot are showing the rank in eigenvalues. The rank is the number of eigenvalues that are not zero. Why are some of the eigenvalues zero? Because of the projectors (nine of them).
                            # 

#%% operator that specifies hpw noise cov should be applied to the fwd
evoked = evokeds[0] # taking the first evoked response.
inv = mne.minimum_norm.make_inverse_operator(evoked.info, fwd, noise_cov) 
                                # now we are doing the forward model and applying that to Martines evoked data to get a time course of the sources.

#%% estimate source time courses for evoked responses
stc = mne.minimum_norm.apply_inverse(evoked, inv, method='MNE')
print(stc.data.shape) # .shape = 8196 (number of sources points), 1201 (time points)
print(src)

stc.plot(subjects_dir=subjects_dir, hemi='both', initial_time=0.170)

#%% load the morph to the template brain - allows for averaging across subjects
morph_name = '0108-oct-6-src-morph.h5' # morphing the data to the template brain (average brain)
morph = mne.read_source_morph(join(bem_path, morph_name)) # now we are projecting Martines brain unto a template brain. And average in that space.

#%%# apply the morph to the subject - bringing them into template space
## this allows for averaging of subjects in source space
stc_morph = morph.apply(stc)
stc_morph.plot(subjects_dir=subjects_dir, hemi='both', initial_time=0.170,
               subject='fsaverage')


#%%# reconstruct individual epochs instead of evoked
stcs = mne.minimum_norm.apply_inverse_epochs(epochs, inv,
                                             lambda2=1, method='MNE',
                                             pick_ori='normal') # pick_ori='normal' = we are not only looking at the amplitude of the signal, but also the direction of the signal.
                                                                # pick_ori='vector' = we are only looking at the direction of the signal.

# These would now be our X: The data that we want to classify on. 

# Mean across epochs - why do we have negative values now as well?
mean_stc = sum(stcs) / len(stcs)
mean_stc.plot(subjects_dir=subjects_dir, hemi='both', initial_time=0.170)



#%% reconstructing single labels

def reconstruct_label(label_name):
    label = mne.read_label(join(bem_path, '..', 'label', label_name))

    stcs = mne.minimum_norm.apply_inverse_epochs(epochs, inv,
                                             lambda2=1, method='MNE',
                                             pick_ori='normal', label=label)

    mean_stc = sum(stcs) / len(stcs) # over trials, not vertices
    return mean_stc

ltc = reconstruct_label('lh.BA44_exvivo.label')
## check the label path for more labels

plt.figure()
plt.plot(ltc.times, ltc.data.T)
plt.show()
