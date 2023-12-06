from sklearn.metrics import confusion_matrix


def informedness(y, yhat):
    """
    Calculates the informedness of a binary classifier.
    """
    tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
    return (tp / (tp + fn)) + (tn / (tn + fp)) - 1
