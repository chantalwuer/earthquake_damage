from setuptools import find_packages
from setuptools import setup

import os

with open('requirements.txt') as f:
    content = f.readlines()
requirements = [x.strip() for x in content if 'git+' not in x]

file = os.path.join(os.path.dirname(__file__), 'eathquake_damage/processed_data/comp_data_household.csv')
setup(name='earthquake_damage',
      description="Project Description",
      packages=find_packages(),
      install_requires=requirements,
      test_suite='tests',
      include_package_data=True,
      data_files=[ file ],
      zip_safe=False
      )
