#! /bin/bash
user=$(uname -n)

if [ $user == 'hyades02' ]
then
    filecontent=( `cat "/projects/Undervisning_CognNeuroSci/scripts/lau/2023/bash/mr_paths.txt" `)
elif [ $user == 'lau' ]
then
    filecontent=( `cat "/home/lau/projects/undervisning_cs/scripts/lau/bash/2023/mr_paths.txt" `)
fi


export SUBJECTS_DIR=/projects/Undervisning_CognNeuroSci/scratch/freesurfer
for path in "${filecontent[@]}"
do
    subject=${path:0:4}
    echo 'Getting labels for SUBJECT: '$subject
    n_threads=1
    submit_to_cluster -w /users/lau/qsubs -q all.q -p MINDLAB2019_MEG-CerebellarClock -n $n_threads "mri_annotation2label --subject $subject --hemi lh --outdir $SUBJECTS_DIR/$subject/label"
    submit_to_cluster -w /users/lau/qsubs -q all.q -p MINDLAB2019_MEG-CerebellarClock -n $n_threads "mri_annotation2label --subject $subject --hemi rh --outdir $SUBJECTS_DIR/$subject/label"
done

