from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

def evaluate_model(model, test_data, test_labels):
    """
    Evaluate the trained model on the test set and print performance metrics.
    """
    model.eval()
    with torch.no_grad():
        test_preds = model(*test_data)

    y_true = test_labels.cpu().numpy()
    y_scores = test_preds.cpu().numpy()
    y_pred = (y_scores > 0.5).astype(int)

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    print("ROC AUC:", roc_auc_score(y_true, y_scores))
    print("PR AUC:", average_precision_score(y_true, y_scores))
