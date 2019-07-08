import numpy as np
import pandas as pd
import scikitplot as skplt
import matplotlib.pyplot as plt
from seqeval.metrics.sequence_labeling import get_entities
from collections import defaultdict


def get_metrics(y_true, y_pred, suffix=False):

    true_entities = set(get_entities(y_true, suffix))
    pred_entities = set(get_entities(y_pred, suffix))

    name_width = 0
    d1 = defaultdict(set)
    d2 = defaultdict(set)
    for e in true_entities:
        d1[e[0]].add((e[1], e[2]))
        name_width = max(name_width, len(e[0]))
    for e in pred_entities:
        d2[e[0]].add((e[1], e[2]))

    type_name_list = []
    ps, rs, f1s, s = [], [], [], []
    for type_name, true_entities in d1.items():
        pred_entities = d2[type_name]
        nb_correct = len(true_entities & pred_entities)
        nb_pred = len(pred_entities)
        nb_true = len(true_entities)

        p = nb_correct / nb_pred if nb_pred > 0 else 0
        r = nb_correct / nb_true if nb_true > 0 else 0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0

        type_name_list.append(type_name)
        ps.append(p)
        rs.append(r)
        f1s.append(f1)
        s.append(nb_true)

    # compute averages
    type_name_list.append('avg / total')
    ps.append(np.average(ps, weights=s))
    rs.append(np.average(rs, weights=s))
    f1s.append(np.average(f1s, weights=s))
    s.append(np.sum(s))

    df_metrics = pd.DataFrame({'type_name': type_name_list, 'precision': ps,
                               'recall': rs, 'f1-score': f1s, 'support': s})

    return df_metrics


def convert_sentence_to_token(y_sentence):
    y_tokens = []
    for y_sen in y_sentence:
        for y_token in y_sen:
            y_tokens.append(y_token)
    return y_tokens


def plot(y_true, y_pred):
    # Confusion matrix
    skplt.metrics.plot_confusion_matrix(convert_sentence_to_token(y_true),
                                        convert_sentence_to_token(y_pred), normalize=True)
    plt.show()

    # Metric
    df_metrics = get_metrics(y_true, y_pred)
    type_name_list = df_metrics['type_name'].tolist()
    ps = df_metrics['precision'].tolist()
    rs = df_metrics['recall'].tolist()
    f1s = df_metrics['f1-score'].tolist()
    s = df_metrics['support'].tolist()

    # Metric F1-Score
    plt.figure(2)
    plt.title('F1-Score')
    plt.bar(range(len(type_name_list)), f1s, tick_label=type_name_list, fc='b')
    for x, y in zip(range(len(type_name_list)), f1s):
        plt.text(x, y, "%0.4f" % y, ha='center', va='bottom')
    plt.ylim([0, 1.1])
    plt.ylabel('F1-Score')
    plt.xlabel('Name Entity Type')
    plt.show()

    # Metric Precision
    plt.figure(3)
    plt.title('Precision')
    plt.bar(range(len(type_name_list)), ps, tick_label=type_name_list, fc='y')
    for x, y in zip(range(len(type_name_list)), ps):
        plt.text(x, y, "%0.4f" % y, ha='center', va='bottom')
    plt.ylim([0, 1.1])
    plt.ylabel('Precision')
    plt.xlabel('Name Entity Type')
    plt.show()

    # Metric Recall
    plt.figure(4)
    plt.title('Recall')
    plt.bar(range(len(type_name_list)), rs, tick_label=type_name_list, fc='y')
    for x, y in zip(range(len(type_name_list)), rs):
        plt.text(x, y, "%0.4f" % y, ha='center', va='bottom')
    plt.ylim([0, 1.1])
    plt.ylabel('Recall')
    plt.xlabel('Name Entity Type')
    plt.show()

    # Metric AllTrueInstanceCnt
    plt.figure(5)
    plt.title('AllTrueInstanceCnt')
    plt.bar(range(len(type_name_list)), s, tick_label=type_name_list, fc='g')
    for x, y in zip(range(len(type_name_list)), s):
        plt.text(x, y, "%d" % y, ha='center', va='bottom')
    plt.ylabel('AllTrueInstanceCnt')
    plt.xlabel('Name Entity Type')
    plt.show()