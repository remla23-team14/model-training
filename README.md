# model-training

Contains the model training pipeline used for sentiment analysis on restaurant reviews. Developed as a part of CS4295 Release Engineering for Machine Learning Applications course.

The repository is structured according to the cookiecutter template.

## 1. Getting started

### 1.1 Requirements

Make sure you have the following dependencies installed:

- Python 3.8
- Poetry 

Navigate https://python-poetry.org/docs/#installation for instructions on how to set up poetry.

### 1.2 Using Poetry



1. Resolving dependencies: Whilst in project directory, run `poetry install` to resolve dependencies
2. Check the python version used with virtual env: `poetry env info`. If needed change python version with `poetry env use python3.8` and re-run step 1.


## 2. DVC Pipeline

The complete model training pipeline and version control for datasets are managed by DVC, and a remote storage has been configured. 

1. Run `poetry run dvc pull` to synchronize the dataset files from the remote repository.
2. Executing `poetry run dvc repro` reproduces the model-training pipeline, which consits of 4 stages: get_data, preprocess, train and evaluate.
3. The model training pipeline produces a model_metrics.json file that stores the accuracy score, precision and recall for the training and test datasets. Use `poetry run dvc metrics show` to view the metrics.
4. `$poetry run dvc metrics diff` shows the change in metrics across an experiment.

NOTE: If you run into missing file errors when doing `poetry run dvc pull`, you might need to do `poetry run dvc fetch` individually for the files. Alternatively, you can try ```poetry run dvc data status --json | jq '.not_in_cache[]' | xargs -L1 -I'{}' poetry run dvc fetch '{}'```


## 3. Code quality

3.1 Pylint

Pylint has been configured to use the DSLinter plugin (https://github.com/SERG-Delft/dslinter). A pylint summary report can be found in `data/reports/pylint.txt`

```
poetry run pylint src
```

3.2 mllint

The code quality is also audited using mllint (https://github.com/bvobart/mllint). An mllint summary report can be found in `data/reports/mllint.txt`

```
poetry run mllint 
```
