# import some functionality
from datetime import datetime

now = datetime.now()
print('Starting script:',now.strftime("%H:%M:%S"))
import os
import pip
import pickle


# Getting back the objects:
N_par=len(models_events) # Number of participants
z_maps_all= np.empty((N_par, 0)).tolist()
conditions_label_all= np.empty((N_par, 0)).tolist()
for i in range(0,21)#range(N_par):
    text = "Loading file %d\n" % (i+1)
    print(text)
    file_name='/work/82777/WordFace_first_level_z_maps_all_trials_all_par_'+str(i)+'.pkl'
    f = open(file_name, 'rb')
    conditions_label_temp, z_maps_temp = pickle.load(f)
    conditions_label_all[i]=conditions_label_temp
    z_maps_all[i]=z_maps_temp
    f.close()

#print(conditions_label_all[9])
            
now = datetime.now()
print('Finishing cell:',now.strftime("%H:%M:%S"))