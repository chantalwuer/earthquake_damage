# Modeling Earthquake Damage in Nepal
This project was done during the final project weeks of the Le Wagon Data Science Bootcamp in December 2022. 
We took data from the earthquake in 2015 in Nepal from the Open Data Portal and performed analyses on it. 
- **Methods:** Data Analysis and Visualisation with Pandas, Matplotlib, Seaborn, Mapping with QGIS, Machine Learning Model with ScikitLearn Keras and Tensorflow
- **Outcome:** Our outcome is a model that is able to predict the expected damage grades of buildings based on their characteristics. Check out the presentation slides in the repository. 
- **Data Source:** Open Data Portal `http://eq2015.npc.gov.np/#/download`

# Install the package

Go to `https://github.com/chantalwuer/earthquake_damage` to see the project, manage issues,
setup you ssh public key, ...

Create a python3 virtualenv and activate it:

```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv -ppython3 ~/venv ; source ~/venv/bin/activate
```

Clone the project and install it:

```bash
git clone git@github.com:chantalwuer/earthquake_damage.git
cd earthquake_damage
pip install -r requirements.txt
make clean install test                # install and test
```
