import os

import numpy as np
import psutil
from sklearn.model_selection import train_test_split

from src.train import train_pipeline

np.random.seed(0)


def test_training_time(benchmark, dataset):
    # Benchmark training time
    seed = 2
    benchmark(train_pipeline, dataset, seed)


def test_memory_usage(dataset):
    seed = 2

    # Measure RSS memory usage before training
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss

    # Train the model
    train_pipeline(dataset, seed)

    # Measure RSS memory usage after training
    final_memory = process.memory_info().rss

    memory_increase = final_memory - initial_memory

    memory_increase_percentage = (memory_increase / initial_memory) * 100
    assert (
        memory_increase_percentage <= 10
    )  # Memory increase should be within the acceptable percentage acceptable percentag


def test_latency(benchmark, dataset, corpus, model, classifier):
    seed = 2

    X = model.fit_transform(corpus).toarray()
    y = dataset.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed
    )
    # Measure the latency of prediction
    benchmark(classifier.predict, X_test)
