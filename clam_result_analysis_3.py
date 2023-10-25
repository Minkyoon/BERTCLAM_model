import pickle
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# 결과 파일들의 경로
result_files = [
    '/home/jsy/2023_clam/results/remission_multimodal_stratified_721_s1/split_0_results.pkl',
    '/home/jsy/2023_clam/results/remission_multimodal_stratified_721_s1/split_1_results.pkl',
    '/home/jsy/2023_clam/results/remission_multimodal_stratified_721_s1/split_2_results.pkl',
    '/home/jsy/2023_clam/results/remission_multimodal_stratified_721_s1/split_3_results.pkl',
    '/home/jsy/2023_clam/results/remission_multimodal_stratified_721_s1/split_4_results.pkl',
    '/home/jsy/2023_clam/results/remission_multimodal_stratified_721_s1/split_5_results.pkl',
    '/home/jsy/2023_clam/results/remission_multimodal_stratified_721_s1/split_6_results.pkl',
    '/home/jsy/2023_clam/results/remission_multimodal_stratified_721_s1/split_7_results.pkl',
    '/home/jsy/2023_clam/results/remission_multimodal_stratified_721_s1/split_8_results.pkl',
    '/home/jsy/2023_clam/results/remission_multimodal_stratified_721_s1/split_9_results.pkl'
]

# 모든 결과를 저장할 리스트
all_true_labels = []
all_predicted_probs = []

# 각 결과 파일에서 데이터를 로드
for file_path in result_files:
    with open(file_path, 'rb') as file:
        results = pickle.load(file)
        all_true_labels.extend([v['label'] for v in results.values()])
        all_predicted_probs.extend([v['prob'][0, 1] for v in results.values()])

# numpy 배열로 변환
all_true_labels = np.array(all_true_labels)
all_predicted_probs = np.array(all_predicted_probs)

threshold = 0.5
all_predicted_labels = (all_predicted_probs >= threshold).astype(int)

# confusion matrix 계산
cm = confusion_matrix(all_true_labels, all_predicted_labels)

# AUROC 계산
fpr, tpr, _ = roc_curve(all_true_labels, all_predicted_probs)
roc_auc = auc(fpr, tpr)

# Sensitivity, Specificity, F1-Score 계산
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
f1 = f1_score(all_true_labels, all_predicted_labels)
recall = sensitivity

# Confusion matrix 그리기
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"verticalalignment": 'center', 'size': 15})
plt.title('Confusion matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks([0.5, 1.5], labels=['Negative', 'Positive'])
plt.yticks([0.5, 1.5], labels=['Negative', 'Positive'], va='center')
plt.show()

# ROC curve 그리기
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# 지표 출력
print(f'Sensitivity: {sensitivity:.2f}')
print(f'Specificity: {specificity:.2f}')
print(f'F1-Score: {f1:.2f}')
print(f'Recall: {recall:.2f}')
