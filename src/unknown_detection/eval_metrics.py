import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import LabelBinarizer

# Set plotting style for publication-ready figures
plt.style.use('seaborn-v0_8-whitegrid')


# Confusion Matricies

# y_true (list/array): The actual ground-truth labels (e.g., ['hand_accel', 'chest_gyro']).
# y_pred (list/array): The labels predicted by the model.
# taxonomy_map (dict, optional): Dictionary mapping specific sensors to broad categories. If None, only Level 2 is plotted.
# level_names (tuple, optional): Titles for the two hierarchy levels. Defaults to ('Modality', 'Sensor Location').
def plot_hierarchical_confusion(y_true, y_pred, taxonomy_map=None, level_names=('Modality', 'Sensor Location')):
    if taxonomy_map is None:
        _plot_cm(y_true, y_pred, title=f"{level_names[1]} Confusion Matrix")
        return

    # Map L2 -> L1
    y_true_l1 = [taxonomy_map.get(label, label) for label in y_true]
    y_pred_l1 = [taxonomy_map.get(label, label) for label in y_pred]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Level 1
    cm_l1 = confusion_matrix(y_true_l1, y_pred_l1)
    labels_l1 = sorted(list(set(y_true_l1) | set(y_pred_l1)))
    
    sns.heatmap(cm_l1, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=labels_l1, yticklabels=labels_l1, ax=axes[0])
    axes[0].set_title(f"Level 1: {level_names[0]} Taxonomy", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Predicted Modality")
    axes[0].set_ylabel("True Modality")

    # Level 2
    cm_l2 = confusion_matrix(y_true, y_pred)
    labels_l2 = sorted(list(set(y_true) | set(y_pred)))
    
    # Rotate labels if there are many classes
    sns.heatmap(cm_l2, annot=True, fmt='d', cmap='Purples', cbar=False,
                xticklabels=labels_l2, yticklabels=labels_l2, ax=axes[1])
    axes[1].set_title(f"Level 2: {level_names[1]} Taxonomy", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Predicted Sensor")
    axes[1].set_ylabel("True Sensor")
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


# ROC Curve

# y_true (list/array): Actual ground-truth labels.
# y_probs (ndarray): 2D array of probabilities from model.predict_proba(). Shape should be (n_samples, n_classes).
# classes (list): The list of class names in the order they appear in y_probs.
def plot_uncertainty_roc(y_true, y_probs, classes):
    y_pred_indices = np.argmax(y_probs, axis=1)
    y_pred_labels = [classes[i] for i in y_pred_indices]
    
    # Confidence
    confidences = np.max(y_probs, axis=1)
    
    # Define "Positive" class as an ERROR
    is_error = (np.array(y_true) != np.array(y_pred_labels)).astype(int)
    uncertainty_scores = 1 - confidences

    # ROC
    fpr, tpr, thresholds = roc_curve(is_error, uncertainty_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'Error Detection AUC = {roc_auc:.2f}')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (Flagging Correct Preds as Uncertain)')
    ax.set_ylabel('True Positive Rate (Detecting Errors)')
    ax.set_title('ROC: Can Confidence Predict Misclassification?', fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    # Youden's J statistic
    J = tpr - fpr
    ix = np.argmax(J)
    best_thresh = thresholds[ix]
    ax.scatter(fpr[ix], tpr[ix], marker='o', color='black', label=f'Best Threshold={best_thresh:.2f}')
    
    plt.tight_layout()
    plt.show()


# ECE

# prob_true (array): Observed frequency of correct predictions per bin.
# prob_pred (array): Mean predicted confidence per bin.
def calculate_ece(prob_true, prob_pred):
    return np.mean(np.abs(prob_true - prob_pred))


# Reliability Curves

# y_true (list/array): Actual ground-truth labels.
# y_probs (ndarray): 2D array of probabilities from model.predict_proba().
# classes (list): Ordered list of class names (from clf.classes_).
# mode (str): 'top-1' evaluates the calibration of the most confident prediction. 'per-class' evaluates calibration for every class individually.
def plot_calibration_curves(y_true, y_probs, classes, mode='top-1'):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")

    if mode == 'top-1':
        confidences = np.max(y_probs, axis=1)
        predictions = np.argmax(y_probs, axis=1)
        pred_labels = [classes[i] for i in predictions]
        
        accuracies = (np.array(y_true) == np.array(pred_labels)).astype(int)
        
        prob_true, prob_pred = calibration_curve(accuracies, confidences, n_bins=10, strategy='uniform')
        ece = calculate_ece(prob_true, prob_pred)
        
        ax.plot(prob_pred, prob_true, "s-", label=f'Top-1 Calibration (ECE={ece:.3f})')       
    elif mode == 'per-class':
        lb = LabelBinarizer()
        lb.fit(classes)
        y_true_bin = lb.transform(y_true)
        
        for i, class_name in enumerate(classes):
            prob_true, prob_pred = calibration_curve(y_true_bin[:, i], y_probs[:, i], n_bins=10)
            ax.plot(prob_pred, prob_true, "s-", alpha=0.6, label=f'{class_name}')

    ax.set_ylabel("Fraction of Positives (Accuracy)")
    ax.set_xlabel("Mean Predicted Value (Confidence)")
    ax.set_title(f"Reliability Diagram ({mode.capitalize()})", fontweight='bold')
    ax.legend(loc="lower right")
    
    plt.tight_layout()
    plt.show()
