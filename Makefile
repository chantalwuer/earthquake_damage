# ----------------------------------
#          INSTALL & TEST
# ----------------------------------

reinstall_package:
	@pip uninstall -y earthquake_damage || :
	@pip install -e .

install:
	@pip install -e .

install_requirements:
	@pip install -r requirements.txt



run_preprocess:
	python -c 'from earthquake_damage.ml_logic.preprocessor import preprocess_features; preprocess_features()'

run_train:
	python -c 'from earthquake_damage.ml_logic.model import train_model; train_model()'

run_pred:
	python -c 'from earthquake_damage.ml_logic.model import predict; predict()'

run_evaluate:
	python -c 'from earthquake_damage.ml_logic.model import cross_validate; cross_validate()'

run_all: run_preprocess run_train run_pred run_evaluate




# ----------------------------------
# 				TEMPLATE
# ----------------------------------

check_code:
	@flake8 scripts/* earthquake_damage/*.py

black:
	@black scripts/* earthquake_damage/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr earthquake_damage-*.dist-info
	@rm -fr earthquake_damage.egg-info

install:
	@pip install . -U

all: clean install test black check_code

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
PYPI_USERNAME=<AUTHOR>
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u $(PYPI_USERNAME)

pypi:
	@twine upload dist/* -u $(PYPI_USERNAME)
