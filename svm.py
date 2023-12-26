import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold

df = pd.read_csv('normalized.csv')
X = df.drop('Label', axis=1)
y = df['Label']


C_values = [0.1, 1, 10, 100]
kernel_types = ['linear', 'poly', 'rbf']

best_params = None
best_mean_accuracy = 0

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for C_value in C_values:
    for kernel_value in kernel_types:
        svm_model = SVC(C=C_value, kernel=kernel_value, decision_function_shape='ovr', probability=True)
        accuracy_scores = cross_val_score(svm_model, X, y, cv=kfold, scoring='accuracy')
        mean_accuracy = np.mean(accuracy_scores)
        print(f"Parameters: C={C_value}, Kernel={kernel_value}")
        print("Accuracy scores for each fold:", accuracy_scores)
        print("Mean accuracy:", mean_accuracy)
        print("-----------------------------")
        if mean_accuracy > best_mean_accuracy:
            best_mean_accuracy = mean_accuracy
            best_params = {'C': C_value, 'kernel': kernel_value}

print("Best Parameters:", best_params)

best_svm_model = SVC(C=best_params['C'], kernel=best_params['kernel'], decision_function_shape='ovr', probability=True)
best_svm_model.fit(X, y)


X_test, _, y_test, _ = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred = best_svm_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print("Test accuracy:", test_accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ROC-AUC, Precision, Recall ,F1-score
roc_auc_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

for class_label in np.unique(y_test):
    y_test_binary = (y_test == class_label).astype(int)
    y_decision_function = best_svm_model.decision_function(X_test)

    y_decision_function_class = y_decision_function[:, np.where(best_svm_model.classes_ == class_label)[0][0]]

    # ROC-AUC
    roc_auc = roc_auc_score(y_test_binary, y_decision_function_class)
    roc_auc_scores.append(roc_auc)
    # Precision, Recall, F1-score
    y_pred_binary = (y_pred == class_label).astype(int)
    precision = precision_score(y_test_binary, y_pred_binary)
    recall = recall_score(y_test_binary, y_pred_binary)
    f1 = f1_score(y_test_binary, y_pred_binary)

    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

mean_roc_auc = np.mean(roc_auc_scores)
mean_precision = np.mean(precision_scores)
mean_recall = np.mean(recall_scores)
mean_f1 = np.mean(f1_scores)

print("Mean ROC-AUC:", mean_roc_auc)
print("Mean Precision:", mean_precision)
print("Mean Recall:", mean_recall)
print("Mean F1-Score:", mean_f1)

#------------------------------------
# ---------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
C_values = [0.1, 1, 10, 100]
kernel_values = ['linear', 'poly', 'rbf']
accuracy_results = []
color_dict = {'linear': 'b', 'poly': 'g', 'rbf': 'r'}

for C_value in C_values:
    for kernel_value in kernel_values:
        svm_model = SVC(C=C_value, kernel=kernel_value, decision_function_shape='ovr', probability=True)
        accuracy_scores = cross_val_score(svm_model, X, y, cv=kfold, scoring='accuracy')
        mean_accuracy = np.mean(accuracy_scores)
        accuracy_results.append((C_value, kernel_value, mean_accuracy))

plt.figure(figsize=(8, 6))
for result in accuracy_results:
    C_value, kernel_value, mean_accuracy = result
    plt.scatter(C_value, mean_accuracy, c=color_dict[kernel_value], label=kernel_value)


plt.xscale('log')
plt.xlabel('C Parameter')
plt.ylabel('Mean Accuracy')
plt.title('K-Fold Cross-Validation Results for SVM')
plt.grid(True)
legend_labels = {'linear': 'Linear Kernel', 'poly': 'Polynomial Kernel', 'rbf': 'RBF Kernel'}
legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_dict[k], markersize=10, label=label) for k, label in legend_labels.items()]
plt.legend(handles=legend_handles, title='Kernel Type')
plt.show()

#---------------------------------------------------------------------------------------------------------------------------------
import seaborn as sns
conf_matrix = confusion_matrix(y_test, y_pred)

# Confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
from sklearn.metrics import roc_curve, auc

# ROC-AUC
plt.figure(figsize=(8, 6))
for class_label in np.unique(y_test):
    y_test_binary = (y_test == class_label).astype(int)
    y_decision_function = best_svm_model.decision_function(X_test)
    y_decision_function_class = y_decision_function[:, np.where(best_svm_model.classes_ == class_label)[0][0]]
    y_decision_function_normalized = (y_decision_function_class - y_decision_function.min()) / (y_decision_function.max() - y_decision_function.min())

    fpr, tpr, _ = roc_curve(y_test_binary, y_decision_function_normalized)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f'Class {class_label} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Each Class')
plt.legend()
plt.show()
#---------------------------------------------------------------------------------------------------------------------------------
from sklearn.metrics import precision_recall_curve
plt.figure(figsize=(8, 6))
for class_label in np.unique(y_test):
    y_test_binary = (y_test == class_label).astype(int)
    y_pred_binary = (y_pred == class_label).astype(int)

    precision, recall, _ = precision_recall_curve(y_test_binary, y_pred_binary)

    plt.plot(recall, precision, label=f'Class {class_label}')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves for Each Class')
plt.legend()
plt.show()

#---------------------------------------------------------------------------------------------------------------------------------
#F1-score
f1_scores_per_class = []
for class_label in np.unique(y_test):
    y_test_binary = (y_test == class_label).astype(int)
    y_pred_binary = (y_pred == class_label).astype(int)

    # F1-score hesapla
    f1 = f1_score(y_test_binary, y_pred_binary)

    # F1-score'ları depolayın
    f1_scores_per_class.append((class_label, f1))


plt.figure(figsize=(8, 6))
for class_label, f1 in f1_scores_per_class:
    plt.scatter(class_label, f1, label=f'Class {class_label}')
plt.xlabel('Class Label')
plt.ylabel('F1-Score')
plt.title('F1-Scores for Each Class')
plt.grid(True)
plt.legend()
plt.show()

#---------------------------------------------------------------------------------------------------------------------------------