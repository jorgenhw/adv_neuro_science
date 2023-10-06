#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 11:38:22 2022

@author: lau
"""

#%% IMPORTS

import mne
from os.path import join
import matplotlib.pyplot as plt
import numpy as np

#%% PREPROCESSING

def preprocess_sensor_space_data(subject, date, raw_path,
                                 h_freq=40,
                                 tmin=-0.200, tmax=1.000, baseline=(None, 0),
                                 reject=None, decim=4):
    recording_names = ['001.self_block1',  '002.other_block1',
                       '003.self_block2',  '004.other_block2',
                       '005.self_block3',  '006.other_block3']
    epochs_list = list()
    for recording_index, recording_name in enumerate(recording_names):
        fif_fname = recording_name[4:]
        full_path = join(raw_path, subject, date, 'MEG', recording_name,
                         'files', fif_fname + '.fif')
        print(full_path)
        raw = mne.io.read_raw(full_path, preload=True)
        raw.filter(l_freq=None, h_freq=h_freq, n_jobs=3)
        
        events = mne.find_events(raw, min_duration=0.002)
        if 'self' in recording_name:
            event_id = dict(self_positive=11, self_negative=12,
                            button_press=23)
        elif 'other' in recording_name: 
            event_id = dict(other_positive=21, other_negative=22,
                            button_press=23)
        else:
            raise NameError('Event codes are not coded for file')
        epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline,
                            preload=True, decim=decim)
        epochs.pick_types(meg=True)
        
        epochs_list.append(epochs)
        
        if recording_index == 0:
            X = epochs.get_data()
            y = epochs.events[:, 2]
        else:
            X = np.concatenate((X, epochs.get_data()), axis=0)
            y = np.concatenate((y, epochs.events[:, 2]))
    
    return epochs_list


def preprocess_source_space_data(subject, date, raw_path, subjects_dir,
                                 epochs_list,
                              method='MNE', lambda2=1, pick_ori='normal',
                              label=None):
    if epochs_list is None:
        epochs_list = preprocess_sensor_space_data(subject, date, raw_path,
                                                   return_epochs=True)
    y = np.zeros(0)
    for epochs in epochs_list: # get y
        y = np.concatenate((y, epochs.events[:, 2]))
    
    if label is not None:
        label_path = join(subjects_dir, subject, 'label', label)
        label = mne.read_label(label_path)
        
    recording_names = ['001.self_block1',  '002.other_block1',
                       '003.self_block2',  '004.other_block2',
                       '005.self_block3',  '006.other_block3']
    for epochs_index, epochs in enumerate(epochs_list): ## get X
        
        fwd_fname = recording_names[epochs_index][4:] + '-oct-6-src-' + \
                    '5120-fwd.fif'
        fwd = mne.read_forward_solution(join(subjects_dir,
                                             subject, 'bem', fwd_fname))
        noise_cov = mne.compute_covariance(epochs, tmax=0.000)
        inv = mne.minimum_norm.make_inverse_operator(epochs.info,
                                                     fwd, noise_cov)
  
        stcs = mne.minimum_norm.apply_inverse_epochs(epochs, inv, lambda2,
                                                     method, label,
                                                     pick_ori=pick_ori)
        for stc_index, stc in enumerate(stcs):
            this_data = stc.data
            if epochs_index == 0 and stc_index == 0:
                n_trials = len(stcs)
                n_vertices, n_samples = this_data.shape
                this_X = np.zeros(shape=(n_trials, n_vertices, n_samples))
            this_X[stc_index, :, :] = this_data
            
        if epochs_index == 0:
            X = this_X
        else:
            X = np.concatenate((X, this_X))
    return X, y

def get_X_and_y(epochs_list):
    for recording_index in range(len(epochs_list)):
        these_epochs = epochs_list[recording_index]
        if recording_index == 0:
            X = these_epochs.get_data()
            y = these_epochs.events[:, 2]
        else:
            X = np.concatenate((X, these_epochs.get_data()), axis=0)
            y = np.concatenate((y, these_epochs.events[:, 2]))
            
    return X, y
#%% RUNNING FUNCTIONS


epochs_list = preprocess_sensor_space_data('0108', '20230928_000000',
        raw_path='/home/lau/analyses/undervisning_cs/raw/',
        decim=10) ##CHANGE TO YOUR PATHS # don't go above decim=10

times = epochs_list[0].times # get time points for later

X_sensor, y = get_X_and_y(epochs_list)

X_source, y = preprocess_source_space_data('0108', '20230928_000000',
        raw_path='/home/lau/analyses/undervisning_cs/raw/', 
        subjects_dir='/home/lau/analyses/undervisning_cs/scratch/freesurfer',
        epochs_list=epochs_list) ##CHANGE TO YOUR PATHS

X_lh_BA44, y = preprocess_source_space_data('0108', '20230928_000000',
        raw_path='/home/lau/analyses/undervisning_cs/raw/', 
        subjects_dir='/home/lau/analyses/undervisning_cs/scratch/freesurfer',
        label='lh.BA44_exvivo.label', epochs_list=epochs_list)
        ##CHANGE TO YOUR PATHS
        
        
X_lh_V1, y = preprocess_source_space_data('0108', '20230928_000000',
        raw_path='/home/lau/analyses/undervisning_cs/raw/', 
        subjects_dir='/home/lau/analyses/undervisning_cs/scratch/freesurfer',
        label='lh.V1_exvivo.label', epochs_list=epochs_list)
        ##CHANGE TO YOUR PATHS        
#%% SIMPLE CLASSIFICATION

def get_indices(y, triggers):
    indices = list()
    for trigger_index, trigger in enumerate(y):
        if trigger in triggers:
            indices.append(trigger_index)
            
    return indices

def equalize_number_of_indices(): # write this yourself
    pass

def simple_classication(X, y, triggers, penalty='none', C=1.0):

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    
    n_samples = X.shape[2]
    indices = get_indices(y, triggers)
    # equalize_number_of_indices()
    X = X[indices, :, :]
    y = y[indices]
    logr = LogisticRegression(penalty=penalty, C=C, solver='newton-cg')
    sc = StandardScaler() # especially necessary for sensor space as
                          ## magnetometers
                          # and gradiometers are on different scales 
                          ## (T and T/m)
    cv = StratifiedKFold()
    
    mean_scores = np.zeros(n_samples)
    
    for sample_index in range(n_samples):
        this_X = X[:, :, sample_index]
        sc.fit(this_X)
        this_X_std = sc.transform(this_X)
        scores = cross_val_score(logr, this_X_std, y, cv=cv)
        mean_scores[sample_index] = np.mean(scores)
        print(sample_index)
        
    return mean_scores

def plot_classfication(times, mean_scores, title=None):

    plt.figure()
    plt.plot(times, mean_scores)
    plt.hlines(0.50, times[0], times[-1], linestyle='dashed', color='k')
    plt.ylabel('Proportion classified correctly')
    plt.xlabel('Time (s)')
    if title is None:
        pass
    else:
        plt.title(title)
    plt.show()

#%% RUN FUNCTION


sensor_pos_neg_self = simple_classication(X_sensor,
                                  y, triggers=[11, 12],
                                  penalty='l2', C=1e-3)

sensor_pos_self_response = simple_classication(X_sensor,
                                  y, triggers=[11, 23],
                                  penalty='l2', C=1e-3) # equalize counts?!

    
#%% PLOT


    
plot_classfication(times, sensor_pos_neg_self,
                   title='positive-self vs. negative-self')

plot_classfication(times, sensor_pos_self_response,
                   title='positive-self vs. response')


#%% COLLAPSE EVENTS (if you want to)

def collapse_events(y, new_value, old_values=list()):
    new_y = y.copy()
    for old_value in old_values:
        new_y[new_y == old_value] = new_value
    return new_y


positive_y = collapse_events(y, 1, [11, 21])
negative_y = collapse_events(y, 2, [12, 22])
self_y =     collapse_events(y, 3, [11, 12])
other_y =    collapse_events(y, 4, [21, 22])

pos_and_neg_y = collapse_events(positive_y, 2, [12, 22])
self_and_other_y = collapse_events(self_y, 4, [21, 22])

#%% CLASSIFCATION - COLLAPSED EVENTS

sensor_pos_neg = simple_classication(X_sensor,
                                  pos_and_neg_y, triggers=[1, 2],
                                  penalty='l2', C=1e-3)

sensor_self_other = simple_classication(X_sensor,
                                  self_and_other_y, triggers=[3, 4],
                                  penalty='l2', C=1e-3)

source_pos_neg = simple_classication(X_source, ## slow 
                                  pos_and_neg_y, triggers=[1, 2],
                                  penalty='l2', C=1e-3)

lh_BA44_self_other = simple_classication(X_lh_BA44, # looking at left BA44 (Broca's area)
                                  self_and_other_y, triggers=[3, 4],
                                  penalty='l2', C=1e-3)

lh_V1_pos_neg = simple_classication(X_lh_V1,
                                  pos_and_neg_y, triggers=[1, 2],
                                  penalty='l2', C=1e-3)
#%% PLOT COLLAPSED

plot_classfication(times, sensor_pos_neg,
                   title='Sensor space: positive vs. negative (self-other collapsed)')


plot_classfication(times, source_pos_neg,
                   title='Source space: positive vs. negative (self-other collapsed)')

plot_classfication(times, sensor_self_other,
                   title='Sensor space: self vs. other (positive-negative collapsed)')

plot_classfication(times, lh_V1_pos_neg,
                   title='left V1: positive vs. negative (self-other collapsed)')

plot_classfication(times, lh_BA44_self_other,
                   title='left BA44: self vs. other (positive-negative collapsed)')

# consider checking if we see the same pattern in the auditory cortex: we should not see the same pattern in the auditory cortex and this could be a kind of sanity check. 
