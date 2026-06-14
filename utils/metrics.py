import torch


def compute_classification_metrics(logits, labels, num_classes):
    preds = torch.argmax(logits, dim=1)
    labels = labels.view(-1)
    correct = (preds == labels).sum().item()
    total = labels.numel()
    wa = correct / total if total else 0.0

    precision_per_class = []
    recall_per_class = []
    f1_per_class = []
    for class_index in range(num_classes):
        pred_pos = preds == class_index
        true_pos = labels == class_index
        tp = (pred_pos & true_pos).sum().item()
        fp = (pred_pos & ~true_pos).sum().item()
        fn = (~pred_pos & true_pos).sum().item()

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precision_per_class.append(precision)
        recall_per_class.append(recall)
        f1_per_class.append(f1)

    macro_f1 = sum(f1_per_class) / num_classes if num_classes else 0.0
    ua = sum(recall_per_class) / num_classes if num_classes else 0.0
    return {
        "accuracy": wa,
        "wa": wa,
        "ua": ua,
        "macro_f1": macro_f1,
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,
        "f1_per_class": f1_per_class,
    }
