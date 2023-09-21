python -m venv virt_env
. /work/798873/virt_env/bin/activate
python -m pip install ipykernel
python -m ipykernel install --user --name=virt_env
echo Done!