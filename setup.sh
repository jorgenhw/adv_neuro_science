# install requirements.txt for project
# Usage: source setup.sh

# create virtual environment
python3 -m venv adv_neuro_sci_venv

# activate virtual environment
source adv_neuro_sci_venv/bin/activate

# Install requirements
python3 -m pip install --upgrade pip

pip3 install -r requirements.txt

# INFO message
echo "Successfully installed requirements.txt"