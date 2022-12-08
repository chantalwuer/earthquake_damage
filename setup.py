from setuptools import find_packages
from setuptools import setup

import os

with open('requirements.txt') as f:
    content = f.readlines()
requirements = [x.strip() for x in content if 'git+' not in x]

setup(name='earthquake_damage',
      description="Project Description",
      packages=find_packages(),
      install_requires=requirements,
      test_suite='tests',
      include_package_data=True,
      data_files=[
          os.path.join(os.path.dirname(__file__), 'processed_data/comp_data_household.csv'),
          os.path.join(os.path.dirname(__file__), 'earthquake_damage/data/preprocessor.pkl'),
          os.path.join(os.path.dirname(__file__), 'Streamlit/fit_best_model.pkl')
      ],
      zip_safe=False
      )
