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

# 각 fold의 ROC curve 정보를 저장할 리스트
tprs = []
mean_fpr = np.linspace(0, 1, 100)
aurocs = []

plt.figure(figsize=(6, 6))

# 각 결과 파일에서 데이터를 로드
for file_path in result_files:
    with open(file_path, 'rb') as file:
        results = pickle.load(file)
        true_labels = [v['label'] for v in results.values()]
        predicted_probs = [v['prob'][0, 1] for v in results.values()]

        # ROC curve 계산
        fpr, tpr, _ = roc_curve(true_labels, predicted_probs)
        roc_auc = auc(fpr, tpr)
        aurocs.append(roc_auc)
        
        # 각 fold의 ROC curve를 그림에 추가
        plt.plot(fpr, tpr, color='navy', alpha=0.1)
        
        # Interpolation을 사용하여 mean_fpr에 대한 tpr 값을 계산
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0

# 평균과 표준편차 계산
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aurocs)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

# 평균 ROC curve와 음영처리된 표준편차 영역을 그림에 추가
plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc))
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2)

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
