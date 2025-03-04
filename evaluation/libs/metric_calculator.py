from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

class MetricsCalculator:
    @staticmethod
    def get_accuracy_metric(true_labels, predictions):
        return accuracy_score(true_labels, predictions)

    @staticmethod
    def get_precision_metric(true_labels, predictions):
        return precision_score(true_labels, predictions)

    @staticmethod
    def get_recall_metric(true_labels, predictions):
        return recall_score(true_labels, predictions)

    @staticmethod
    def get_f1_score_metric(true_labels, predictions):
        return f1_score(true_labels, predictions)

    @staticmethod
    def get_confusion_metric(true_labels, predictions):
        return confusion_matrix(true_labels, predictions)

    @staticmethod
    def get_roc_auc_score_metric(true_labels, probabilities):
        return roc_auc_score(true_labels, probabilities)

    @staticmethod
    def calculate_all_metrics(true_labels, predictions, probabilities):
        return {
            "accuracy": MetricsCalculator.get_accuracy_metric(true_labels, predictions),
            "precision": MetricsCalculator.get_precision_metric(true_labels, predictions),
            "recall": MetricsCalculator.get_recall_metric(true_labels, predictions),
            "f1": MetricsCalculator.get_f1_score_metric(true_labels, predictions),
            "confusion_matrix": MetricsCalculator.get_confusion_metric(true_labels, predictions),
            "auc_roc": MetricsCalculator.get_roc_auc_score_metric(true_labels, probabilities),
        }

