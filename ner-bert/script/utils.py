import numpy as np
import pandas as pd
import scikitplot as skplt
import matplotlib.pyplot as plt
import os
import json
from seqeval.metrics.sequence_labeling import get_entities
from collections import defaultdict
from azureml.core.run import Run


def serialize_result(raw_text_list, y_list):
    result_list = []
    for i, y in enumerate(y_list):
        text_list = raw_text_list[i].split(' ')
        entities = get_entities(y)
        result = []
        for e in entities:
            start_index = e[1]
            end_index = e[2] + 1
            entity_name = ' '.join(text_list[start_index: end_index])
            entity_label = e[0]
            result.append(tuple((entity_name, entity_label)))
        result_dict = {name: {"tag": label} for (name, label) in result}
        # result_json = json.dumps({name: {"tag": label} for (name, label) in result})
        result_list.append(result_dict)
    return pd.DataFrame({'Text': raw_text_list, 'PredictedLabel': result_list})


def deserialize_result(result_list):
    entities = set()
    for result in result_list:
        result_dict = json.loads(result)
        for name in result_dict:
            entities.add(tuple((result_dict[name]["tag"], name)))
    return entities


def save_as_df(output_df, output_eval_dir):
    if output_eval_dir != '':
        if not os.path.exists(output_eval_dir):
            os.makedirs(output_eval_dir)
        pred_label_list = output_df['PredictedLabel'].tolist()
        final_df = pd.DataFrame({'Text': output_df['Text'].tolist(), 'PredictedLabel': [json.dumps(p_dict) for p_dict in pred_label_list]})
        print(final_df)
        final_df.to_parquet(fname=os.path.join(output_eval_dir, "prediction.parquet"), engine='pyarrow')
        

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


def plot(y_true, y_pred, output_eval_dir):
    run = Run.get_context()
    # # Confusion matrix
    # skplt.metrics.plot_confusion_matrix(convert_sentence_to_token(y_true),
    #                                     convert_sentence_to_token(y_pred), normalize=True)
    # run.log_image("metrics/confusion_matrix", plot=plt)
    # plt.savefig(os.path.join(output_eval_dir, 'confusion_matrix.png'))
    # # plt.show()

    # Metric
    df_metrics = get_metrics(y_true, y_pred)
    type_name_list = df_metrics['type_name'].tolist()
    ps = df_metrics['precision'].tolist()
    rs = df_metrics['recall'].tolist()
    f1s = df_metrics['f1-score'].tolist()
    s = df_metrics['support'].tolist()

    # Metric F1-Score
    f1_plt = plt.figure(2)
    plt.title('F1-Score')
    plt.bar(range(len(type_name_list)), f1s, tick_label=type_name_list, fc='b')
    for x, y in zip(range(len(type_name_list)), f1s):
        plt.text(x, y, "%0.4f" % y, ha='center', va='bottom')
    plt.ylim([0, 1.1])
    plt.ylabel('F1-Score')
    plt.xlabel('Name Entity Type')
    run.log_image("metrics/f1_score", plot=f1_plt)
    f1_plt.savefig(os.path.join(output_eval_dir, 'f1_score.png'))
    # plt.show()

    # Metric Precision
    precision_plt = plt.figure(3)
    plt.title('Precision')
    plt.bar(range(len(type_name_list)), ps, tick_label=type_name_list, fc='y')
    for x, y in zip(range(len(type_name_list)), ps):
        plt.text(x, y, "%0.4f" % y, ha='center', va='bottom')
    plt.ylim([0, 1.1])
    plt.ylabel('Precision')
    plt.xlabel('Name Entity Type')
    run.log_image("metrics/precision", plot=precision_plt)
    precision_plt.savefig(os.path.join(output_eval_dir, 'precision.png'))
    # plt.show()

    # Metric Recall
    recall_plt = plt.figure(4)
    plt.title('Recall')
    plt.bar(range(len(type_name_list)), rs, tick_label=type_name_list, fc='y')
    for x, y in zip(range(len(type_name_list)), rs):
        plt.text(x, y, "%0.4f" % y, ha='center', va='bottom')
    plt.ylim([0, 1.1])
    plt.ylabel('Recall')
    plt.xlabel('Name Entity Type')
    run.log_image("metrics/recall", plot=recall_plt)
    recall_plt.savefig(os.path.join(output_eval_dir, 'recall.png'))
    # plt.show()

    # Metric AllTrueInstanceCnt
    gt_plt = plt.figure(5)
    plt.title('AllTrueInstanceCnt')
    plt.bar(range(len(type_name_list)), s, tick_label=type_name_list, fc='g')
    for x, y in zip(range(len(type_name_list)), s):
        plt.text(x, y, "%d" % y, ha='center', va='bottom')
    plt.ylabel('AllTrueInstanceCnt')
    plt.xlabel('Name Entity Type')
    run.log_image("metrics/ground_truth", plot=gt_plt)
    gt_plt.savefig(os.path.join(output_eval_dir, 'ground_truth.png'))
    # plt.show()