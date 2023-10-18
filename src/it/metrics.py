import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)


def plot_confusion_matrix(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)
    ax = plt.subplot()
    sns.heatmap(conf_matrix, annot=True, ax=ax, cmap="Blues", fmt="g", cbar=False)

    # Add labels, title and ticks
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Acctual")
    ax.set_title("Confusion Matrix")
    ax.xaxis.set_ticklabels(["Benign", "Attack"])
    ax.yaxis.set_ticklabels(["Benign", "Attack"])


def print_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    print(f"Accuracy: {acc:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")


def per_class_accuracy(y_true, y_pred):
    cmd = confusion_matrix(y_true, y_pred, normalize="true").diagonal()
    per_class_accuracy_df = pd.DataFrame(
        [(index, round(value, 4)) for index, value in zip(["Benign", "Attack"], cmd)],
        columns=["type", "accuracy"],
    )
    per_class_accuracy_df = per_class_accuracy_df.round(2)
    return per_class_accuracy_df
