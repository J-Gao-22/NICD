import numpy as np
from sklearn.metrics import roc_auc_score as sklearn_auc

def roc_auc_score(y, s, *args, **kwargs):
    if np.unique(y).size < 2:
        raise ValueError('Only one class present in y_true')
    return sklearn_auc(y, s, *args, **kwargs)

def reorganize_by_user(user, predict, label):
    if not isinstance(predict, np.ndarray):
        predict = np.array(predict)
    if not isinstance(label, np.ndarray):
        label = np.array(label)
    predict = predict.squeeze()
    label = label.squeeze()
    u, inverse, counts = np.unique(user, return_inverse=True, return_counts=True)
    index = np.argsort(inverse)
    candidates_dict = {}
    discard_dict = {}
    k = 0
    total_num = 0
    only_one_interaction = 0
    for u_i in u:
        start_id, end_id = (total_num, total_num + counts[k])
        u_i_counts = counts[k]
        index_ui = index[start_id:end_id]
        if u_i_counts == 1:
            only_one_interaction += 1
            total_num += counts[k]
            k += 1
            discard_dict[u_i] = [predict[index_ui], label[index_ui]]
            continue
        candidates_dict[u_i] = [predict[index_ui], label[index_ui]]
        total_num += counts[k]
        k += 1
    print('only one interaction users:', only_one_interaction)
    return candidates_dict

def uAUC_me(candidates_dict):
    computed_u = []
    auc = []
    only_one_class = 0
    for ui, pre_and_true in candidates_dict.items():
        pre_i, label_i = pre_and_true
        try:
            ui_auc = roc_auc_score(label_i, pre_i)
            auc.append(ui_auc)
            computed_u.append(ui)
        except:
            only_one_class += 1
    auc_for_user = np.array(auc)
    print('computed user:', auc_for_user.shape[0], 'can not users:', only_one_class)
    uauc = auc_for_user.mean()
    return (uauc, computed_u, auc_for_user)

def hr_at_k(candidates_dict, k):
    hr_list = []
    for uid, pre_and_true in candidates_dict.items():
        pred, label = pre_and_true
        if len(pred) == 0:
            continue
        top_k_indices = np.argsort(-np.array(pred))[:k]
        hit = np.any(np.array(label)[top_k_indices] == 1)
        hr_list.append(hit)
    return np.mean(hr_list) if hr_list else 0.0

def ndcg_at_k(candidates_dict, k):
    ndcg_list = []
    for uid, pre_and_true in candidates_dict.items():
        pred, label = pre_and_true
        if len(pred) == 0:
            continue
        ranked_labels = np.array(label)[np.argsort(-np.array(pred))]
        dcg = 0.0
        for i in range(min(k, len(ranked_labels))):
            rel = ranked_labels[i]
            dcg += rel / np.log2(i + 2)
        ideal_labels = np.sort(np.array(label))[::-1]
        idcg = 0.0
        for i in range(min(k, len(ideal_labels))):
            rel = ideal_labels[i]
            idcg += rel / np.log2(i + 2)
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_list.append(ndcg)
    return np.mean(ndcg_list) if ndcg_list else 0.0

def map_at_k(candidates_dict, k):
    ap_list = []
    for uid, pre_and_true in candidates_dict.items():
        pred, label = pre_and_true
        if len(pred) == 0:
            continue
        ranked_labels = np.array(label)[np.argsort(-np.array(pred))]
        hits = 0
        sum_precision = 0.0
        for i in range(min(k, len(ranked_labels))):
            if ranked_labels[i] == 1:
                hits += 1
                precision_at_i = hits / (i + 1)
                sum_precision += precision_at_i
        ap = sum_precision / np.sum(label) if np.sum(label) > 0 else 0.0
        ap_list.append(ap)
    return np.mean(ap_list) if ap_list else 0.0
