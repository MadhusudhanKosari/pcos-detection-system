import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

# Load probabilities
clinical_prob = np.load("outputs/clinical_probabilities.npy")
clinical_y = np.load("outputs/clinical_true_labels.npy")

image_prob = np.load("outputs/image_probabilities.npy")

print("Clinical probs:", clinical_prob.shape)
print("Image probs:", image_prob.shape)

# Since datasets are unaligned, we use MEAN image probability
mean_image_prob = np.mean(image_prob)

print("Mean Image Probability:", mean_image_prob)

# Try different fusion weights
alphas = [0.9, 0.8, 0.7, 0.6, 0.5]

print("\nDecision-Level Fusion Results:\n")

for alpha in alphas:
    fused_prob = alpha * clinical_prob + (1 - alpha) * mean_image_prob
    fused_pred = (fused_prob >= 0.5).astype(int)

    acc = accuracy_score(clinical_y, fused_pred)
    auc = roc_auc_score(clinical_y, fused_prob)

    print(f"Alpha = {alpha}")
    print(f"  Accuracy: {acc:.3f}")
    print(f"  ROC-AUC:  {auc:.3f}")
