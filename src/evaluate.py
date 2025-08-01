from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_class = (y_pred > 0.5).astype(int)

    print(confusion_matrix(y_test, y_pred_class))
    print(classification_report(y_test, y_pred_class))
    print("ROC AUC:", roc_auc_score(y_test, y_pred))
