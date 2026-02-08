import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def extract_ddr4cc_features(img_path):

    try:
        img_array = np.fromfile(img_path, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    except:
        return None
    if img is None: return None

    R, C = img.shape
    v_acc = np.sum(img, axis=0) / (255.0 * R)
    h_acc = np.sum(img, axis=1) / (255.0 * C)

    maxV = np.max(v_acc)
    maxH = np.max(h_acc)
    dV = maxV - np.mean(v_acc)
    dH = maxH - np.mean(h_acc)
    return [maxV, dV, maxH, dH]


def run_classifier_test(index_file, img_dir):
    data, labels = [], []
    l_map = {'G-Longitudinal': 0, 'G-Transverse': 1, 'G-Alligator': 2, 'G-Healthy': 3}

    if not os.path.exists(index_file): return None

    with open(index_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2: continue
            feat = extract_ddr4cc_features(os.path.join(img_dir, parts[0]))
            if feat:
                data.append(feat)
                labels.append(l_map[parts[1]])

    X, y = np.array(data), np.array(labels)

    models = {
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "BFTree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "PART": DecisionTreeClassifier(criterion='entropy', max_depth=6, min_samples_leaf=2, random_state=42),
        "SVM (Linear)": make_pipeline(StandardScaler(), SVC(kernel='linear', C=0.05, random_state=42))
    }
    scores_dict = {}
    for name, clf in models.items():
        try:
            cv_scores = cross_val_score(clf, X, y, cv=10, scoring='f1_weighted')
            scores_dict[name] = round(cv_scores.mean(), 4)
        except:
            scores_dict[name] = 0.0
    return scores_dict
if __name__ == "__main__":
    gt_dir = r"E:Combined_GT_Images"
    mts_dir = r"E:Combined_MTS_Images"
    gt_index = "Combined_GT_index.txt"
    mts_index = "Combined_MTS_index.txt"

    print("Running classification comparison experiments....")
    res_gt = run_classifier_test(gt_index, gt_dir)
    res_mts = run_classifier_test(mts_index, mts_dir)

    if res_gt and res_mts:
        df = pd.DataFrame([res_gt, res_mts], index=['DDR4CC on GT Mask', 'DDR4CC on MTSCrack Mask'])
        print(df)
        df.to_csv("DDR4CC_Final_Comparison.csv")