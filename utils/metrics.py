import torch


def compute_confusion_matrix(preds, labels, num_classes):
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for true_label, pred_label in zip(labels.view(-1), preds.view(-1)):
        matrix[int(true_label), int(pred_label)] += 1
    return matrix


def compute_classification_metrics(logits, labels, num_classes, label_names=None):
    preds = torch.argmax(logits, dim=1)
    labels = labels.view(-1)
    correct = (preds == labels).sum().item()
    total = labels.numel()
    wa = correct / total if total else 0.0
    confusion_matrix = compute_confusion_matrix(preds, labels, num_classes)

    precision_per_class = []
    recall_per_class = []
    f1_per_class = []
    per_class = []
    for class_index in range(num_classes):
        tp = confusion_matrix[class_index, class_index].item()
        fp = confusion_matrix[:, class_index].sum().item() - tp
        fn = confusion_matrix[class_index, :].sum().item() - tp

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precision_per_class.append(precision)
        recall_per_class.append(recall)
        f1_per_class.append(f1)
        class_name = label_names[class_index] if label_names and class_index < len(label_names) else str(class_index)
        per_class.append(
            {
                "class_index": class_index,
                "class_name": class_name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": confusion_matrix[class_index, :].sum().item(),
            }
        )

    macro_f1 = sum(f1_per_class) / num_classes if num_classes else 0.0
    ua = sum(recall_per_class) / num_classes if num_classes else 0.0
    return {
        "accuracy": wa,
        "wa": wa,
        "ua": ua,
        "macro_f1": macro_f1,
        "confusion_matrix": confusion_matrix.tolist(),
        "per_class": per_class,
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,
        "f1_per_class": f1_per_class,
    }
