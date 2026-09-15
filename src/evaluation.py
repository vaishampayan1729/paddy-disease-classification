import torch

#===================================================
# Function to get predictions from the model
#===================================================
def get_predictions(model, data_loader, device):
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            predictions = logits.argmax(dim=1)

            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())

    predictions = torch.cat(all_predictions)
    labels = torch.cat(all_labels)

    return predictions, labels

#===================================================
# Function to calculate accuracy
#===================================================
def calculate_accuracy(predictions, labels):
    correct = (predictions == labels).sum().item()
    total = labels.numel()

    return correct / total


#===================================================
# Function to calculate confusion matrix
#===================================================
def calculate_confusion_matrix(
    predictions,
    labels,
    num_classes
):
    confusion_matrix = torch.zeros(
        num_classes,
        num_classes,
        dtype=torch.int64
    )

    for true_label, predicted_label in zip(labels, predictions):
        confusion_matrix[true_label, predicted_label] += 1

    return confusion_matrix

#===================================================
# Function to return individual F1 scores
#===================================================
def calculate_f1_scores(predictions, labels, num_classes):
    confusion_matrix = calculate_confusion_matrix(
        predictions,
        labels,
        num_classes
    )

    true_positives = confusion_matrix.diag()

    false_positives = (
        confusion_matrix.sum(dim=0) - true_positives
    )

    false_negatives = (
        confusion_matrix.sum(dim=1) - true_positives
    )

    precision = true_positives / (
        true_positives + false_positives
    ).clamp(min=1)

    recall = true_positives / (
        true_positives + false_negatives
    ).clamp(min=1)

    f1_scores = 2 * precision * recall / (
        precision + recall
    ).clamp(min=1e-8)

    return f1_scores

#===================================================
# Function to evaluate macro F1 score
#===================================================
def calculate_macro_f1(predictions, labels, num_classes):
    f1_scores = calculate_f1_scores(
        predictions,
        labels,
        num_classes
    )

    return f1_scores.mean().item()