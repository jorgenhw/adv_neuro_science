#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 09:26:42 2022

@author: lau
"""

subjects = [
            # '0108', 
            # '0109', '0110',
            # '0111', '0112', '0113',
            '0114', '0115'
           ]
dates = [
         # '20230928_000000',
          # '20230926_000000', '20230926_000000',
           # '20230926_000000', '20230927_000000', '20230927_000000',
          '20230927_000000', '20230928_000000'
         ]

subjects_dir = '/home/lau//projects/undervisning_cs/scratch/freesurfer'
raw_path = '/home/lau//projects/undervisning_cs/raw'

import mne
from os.path import join

#%% WATERSHED

for subject in subjects:
    mne.bem.make_watershed_bem(subject, subjects_dir)
    
#%% CORTICAL SURFACES

for subject in subjects:
    src = mne.source_space.setup_source_space(subject,
                                              subjects_dir=subjects_dir,
                                              n_jobs=7)
    bem_path = join(subjects_dir, subject, 'bem')
    write_filename = subject + '-oct-6-src.fif'
    mne.source_space.write_source_spaces(join(bem_path, write_filename), src)

#%% MORPH TO FSAVERAGE

for subject in subjects:
    bem_path = join(subjects_dir, subject, 'bem')
    read_filename = subject + '-oct-6-src.fif'
    write_filename = subject + '-oct-6-src-morph.h5'

    src = mne.source_space.read_source_spaces(join(bem_path, read_filename))
    
    morph = mne.compute_source_morph(src, subject, subjects_dir=subjects_dir)
    morph.save(join(bem_path, write_filename))
  
#%% MAKE SCALP SURFACES

for subject in subjects:
    mne.bem.make_scalp_surfaces(subject, subjects_dir)    
    
#%% BEM MODEL

for subject in subjects:
    bem_path = join(subjects_dir, subject, 'bem')

    ## single-layer model
    write_filename = subject + '-5120-bem.fif'
    bem_surfaces = mne.bem.make_bem_model(subject, conductivity=[0.3],
                                          subjects_dir=subjects_dir)
    mne.bem.write_bem_surfaces(join(bem_path, write_filename),
                               bem_surfaces)
        
#%% BEM SOLUTION

for subject in subjects:
    bem_path = join(subjects_dir, subject, 'bem')
    ## single-layer
    read_filename = subject + '-5120-bem.fif'
    write_filename = subject + '-5120-bem-sol.fif'
    bem_surfaces = mne.bem.read_bem_surfaces(join(bem_path, read_filename))
    bem_solution = mne.bem.make_bem_solution(bem_surfaces)
    mne.bem.write_bem_solution(join(bem_path, write_filename),
                               bem_solution)

#%% FORWARD MODELS
for subject, date in zip(subjects, dates):
    bem_path = join(subjects_dir, subject, 'bem')
    trans = join(bem_path, subject + '-trans.fif')
    src = join(bem_path, subject + '-oct-6-src.fif')
    recording_names = ['001.self_block1',  '002.other_block1',
                       '003.self_block2',  '004.other_block2',
                       '005.self_block3',  '006.other_block3']
   
    for recording_index in range(len(recording_names)):
        read_filename = recording_names[recording_index]
        fif_fname = read_filename[4:]
        meg_path = join(raw_path, subject, date,
                        'MEG', read_filename, 'files',
                        fif_fname + '.fif')
        info = mne.io.read_info(meg_path)
        bem = join(bem_path, subject + '-5120-bem-sol.fif')
        fwd = mne.make_forward_solution(info, trans, src, bem, n_jobs=7)
        
        write_filename = fif_fname + '-oct-6-src-5120-fwd.fif' 
        print(write_filename)
        mne.write_forward_solution(join(bem_path, write_filename), fwd,
                                   overwrite=True)   