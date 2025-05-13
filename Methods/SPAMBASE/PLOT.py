import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import roc_curve, auc

# Define the methods and assign distinct colors
#methods = ['SMOTE', 'GAN', 'SmotifiedGAN', 'MCMC']
#colors = ['blue', 'green', 'red', 'purple']  # Different colors for each method


# Define the methods and assign distinct colors
methods = ['NON','SMOTE', 'GAN','SMOTifiedGAN', 'MCMC','MCMCGAN']
models = ['NON-Oversampled','SMOTE', 'GAN','SMOTified-GAN', 'MCMC','MCMC-GAN']
colors = ['green', 'blue', 'red','purple','pink','orange']  # Different colors for each method
linestyles = ['-', '--', ':', '-.', '-', '--']  # Different line styles
markers = ['o', '', '^', '', 'v', '']  # Different markers
#markers = ['o', 's', '^', 'd', 'v', '*']  # Different markers
plt.figure(figsize=(20, 10))

for method, color, linestyle,models in zip(methods, colors, linestyles,models):
    with open(f'SPAMBASE_{method}.pkl', 'rb') as f:
        precision, recall, pr_auc = pickle.load(f)
    
    #plt.plot(recall, precision, color=color, label=f'{method} (AUC={pr_auc:.3f})')
    plt.plot(recall, precision, linestyle=linestyle, color=color, 
            alpha=0.8, label=f'{models} (AUC={pr_auc:.3f})')


# Graph Labels
plt.xlabel('Recall', fontsize=20)
plt.ylabel('Precision', fontsize=20)
#plt.title('Precision-Recall Curve Comparison', fontsize=14)
plt.legend(loc='best', fontsize=20)
plt.grid(alpha=0.3)
plt.tight_layout()

# Show the plot
plt.show()




# # Plot ROC curve (AUC)
# plt.figure(figsize=(8, 6))
# for method, color in zip(methods, colors):
#     # Load the predicted probabilities (ypr) for each method from the pickle file
#     with open(f'predictions_{method}.pkl', 'rb') as f:
#         y_true, y_pred_prob = pickle.load(f)  # Assuming 'y_true' is the true labels and 'y_pred_prob' is the predicted probabilities

#     # Compute ROC curve and AUC
#     fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
#     roc_auc = auc(fpr, tpr)
    
#     plt.plot(fpr, tpr, color=color, label=f'{method} (AUC={roc_auc:.3f})')

# # Graph Labels for ROC curve
# plt.xlabel('False Positive Rate', fontsize=12)
# plt.ylabel('True Positive Rate', fontsize=12)
# plt.title('Receiver Operating Characteristic (ROC) Curve Comparison', fontsize=14)
# plt.legend(loc='best', fontsize=10)
# plt.grid(alpha=0.3)

# # Show the plot for ROC curve
# plt.tight_layout()
# plt.show()