from src.evaluate import eval_score


def test_positive_data_slice(dataset, corpus, model, classifier):
    # Filter the corpus based on positive words
    positive_words = ["good", "glad", "great", "satisfied"]
    positive_indices = [
        i
        for i, review in enumerate(corpus)
        if any(word in review for word in positive_words)
    ]
    positive_corpus = [corpus[i] for i in positive_indices]

    # Slice the dataset based on positive sentiment
    positive_reviews = dataset.iloc[positive_indices]

    # Assert that the positive sentiment slice has non-empty reviews
    assert len(positive_reviews) > 0
    assert len(positive_corpus) == len(positive_reviews)

    # Calculate metrics for the complete data
    X = model.fit_transform(corpus).toarray()
    y = dataset.iloc[:, -1].values
    complete_metrics = eval_score(classifier, X, y)

    # Calculate metrics for the positive data slice
    X_positive = X[positive_indices]
    Y_positive = y[positive_indices]
    positive_metrics = eval_score(classifier, X_positive, Y_positive)

    # Compare the metrics of the positive data slice with the metrics of the complete data
    assert (
        positive_metrics[0] >= complete_metrics[0] * 0.9
    )  # Accuracy threshold (90% of complete accuracy)
    assert (
        positive_metrics[1] >= complete_metrics[1] * 0.9
    )  # Precision threshold (90% of complete precision)
    assert (
        positive_metrics[2] >= complete_metrics[2] * 0.9
    )  # Recall threshold (90% of complete recall)

def test_negated_words_data_slice(dataset, corpus, model, classifier):   
    # Filter the corpus based on negated words
    negated_words = ["isn't", "not", "never", "didn't", "wouldn't", "don't", "won't"]
    negated_indices = [i for i, review in enumerate(corpus) if any(word in review for word in negated_words)]
    negated_corpus = [corpus[i] for i in negated_indices]
    
    # Slice the dataset based on negated words
    negated_reviews = dataset.iloc[negated_indices]
    
    # Assert that the negated words slice has non-empty reviews
    assert len(negated_reviews) > 0
    assert len(negated_corpus) == len(negated_reviews)
    
    # Calculate metrics for the complete data
    X = model.fit_transform(corpus).toarray()
    y = dataset.iloc[:, -1].values
    complete_metrics = eval_score(classifier, X, y)
    
    # Calculate metrics for the negated words data slice
    X_negated = X[negated_indices]
    Y_negated = y[negated_indices]
    negated_metrics = eval_score(classifier, X_negated, Y_negated)
    
    # Compare the metrics of the negated words data slice with the metrics of the complete data
    assert negated_metrics[0] >= complete_metrics[0] * 0.9  # Accuracy threshold (90% of complete accuracy)
    assert negated_metrics[1] >= complete_metrics[1] * 0.9  # Precision threshold (90% of complete precision)
    assert negated_metrics[2] >= complete_metrics[2] * 0.8  # Recall threshold (90% of complete recall)