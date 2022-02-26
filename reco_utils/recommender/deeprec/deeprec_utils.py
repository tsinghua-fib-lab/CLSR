# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import os
import six
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    mean_squared_error,
    accuracy_score,
    f1_score,
)
import numpy as np
import pandas as pd
import yaml
import zipfile
import json
import pickle as pkl
import tensorflow as tf

from reco_utils.dataset.download_utils import maybe_download


def flat_config(config):
    """Flat config loaded from a yaml file to a flat dict.
    
    Args:
        config (dict): Configuration loaded from a yaml file.

    Returns:
        dict: Configuration dictionary.
    """
    f_config = {}
    category = config.keys()
    for cate in category:
        for key, val in config[cate].items():
            f_config[key] = val
    return f_config


def check_type(config):
    """Check that the config parameters are the correct type
    
    Args:
        config (dict): Configuration dictionary.

    Raises:
        TypeError: If the parameters are not the correct type.
    """

    int_parameters = [
        "word_size",
        "entity_size",
        "doc_size",
        "history_size",
        "FEATURE_COUNT",
        "FIELD_COUNT",
        "dim",
        "epochs",
        "batch_size",
        "show_step",
        "save_epoch",
        "PAIR_NUM",
        "DNN_FIELD_NUM",
        "attention_layer_sizes",
        "n_user",
        "n_item",
        "n_user_attr",
        "n_item_attr",
        "item_embedding_dim",
        "cate_embedding_dim",
        "user_embedding_dim",
        "max_seq_length",
        "hidden_size",
        "T",
        "L",
        "n_v",
        "n_h",
        "kernel_size",
        "min_seq_length",
        "attention_size",
        "epochs",
        "batch_size",
        "show_step",
        "save_epoch",
        "train_num_ngs",
    ]
    for param in int_parameters:
        if param in config and not isinstance(config[param], int):
            raise TypeError("Parameters {0} must be int".format(param))

    float_parameters = [
        "init_value",
        "learning_rate",
        "embed_l2",
        "embed_l1",
        "layer_l2",
        "layer_l1",
        "mu",
    ]
    for param in float_parameters:
        if param in config and not isinstance(config[param], float):
            raise TypeError("Parameters {0} must be float".format(param))

    str_parameters = [
        "train_file",
        "eval_file",
        "test_file",
        "infer_file",
        "method",
        "load_model_name",
        "infer_model_name",
        "loss",
        "optimizer",
        "init_method",
        "attention_activation",
        "user_vocab",
        "item_vocab",
        "cate_vocab",
    ]
    for param in str_parameters:
        if param in config and not isinstance(config[param], str):
            raise TypeError("Parameters {0} must be str".format(param))

    list_parameters = [
        "layer_sizes",
        "activation",
        "dropout",
        "att_fcn_layer_sizes",
        "dilations",
    ]
    for param in list_parameters:
        if param in config and not isinstance(config[param], list):
            raise TypeError("Parameters {0} must be list".format(param))


def check_nn_config(f_config):
    """Check neural networks configuration.
    
    Args:
        f_config (dict): Neural network configuration.
    
    Raises:
        ValueError: If the parameters are not correct.
    """
    if f_config["model_type"] in ["fm", "FM"]:
        required_parameters = ["FEATURE_COUNT", "dim", "loss", "data_format", "method"]
    elif f_config["model_type"] in ["lr", "LR"]:
        required_parameters = ["FEATURE_COUNT", "loss", "data_format", "method"]
    elif f_config["model_type"] in ["dkn", "DKN"]:
        required_parameters = [
            "doc_size",
            "history_size",
            "wordEmb_file",
            "entityEmb_file",
            "contextEmb_file",
            "news_feature_file",
            "user_history_file",
            "word_size",
            "entity_size",
            "use_entity",
            "use_context",
            "data_format",
            "dim",
            "layer_sizes",
            "activation",
            "attention_activation",
            "attention_activation",
            "attention_dropout",
            "loss",
            "data_format",
            "dropout",
            "method",
            "num_filters",
            "filter_sizes",
        ]
    elif f_config["model_type"] in ["exDeepFM", "xDeepFM"]:
        required_parameters = [
            "FIELD_COUNT",
            "FEATURE_COUNT",
            "method",
            "dim",
            "layer_sizes",
            "cross_layer_sizes",
            "activation",
            "loss",
            "data_format",
            "dropout",
        ]
    if f_config["model_type"] in ["gru4rec", "GRU4REC", "GRU4Rec"]:
        required_parameters = [
            "item_embedding_dim",
            "cate_embedding_dim",
            "max_seq_length",
            "loss",
            "method",
            "user_vocab",
            "item_vocab",
            "cate_vocab",
            "hidden_size",
        ]
    elif f_config["model_type"] in ["caser", "CASER", "Caser"]:
        required_parameters = [
            "item_embedding_dim",
            "cate_embedding_dim",
            "user_embedding_dim",
            "max_seq_length",
            "loss",
            "method",
            "user_vocab",
            "item_vocab",
            "cate_vocab",
            "T",
            "L",
            "n_v",
            "n_h",
            "min_seq_length",
        ]
    elif f_config["model_type"] in ["asvd", "ASVD", "a2svd", "A2SVD"]:
        required_parameters = [
            "item_embedding_dim",
            "cate_embedding_dim",
            "max_seq_length",
            "loss",
            "method",
            "user_vocab",
            "item_vocab",
            "cate_vocab",
        ]
    elif f_config["model_type"] in ["slirec", "sli_rec", "SLI_REC", "Sli_rec"]:
        required_parameters = [
            "item_embedding_dim",
            "cate_embedding_dim",
            "max_seq_length",
            "loss",
            "method",
            "user_vocab",
            "item_vocab",
            "cate_vocab",
            "attention_size",
            "hidden_size",
            "att_fcn_layer_sizes",
        ]
    elif f_config["model_type"] in ["clsr", "CLSR"]:
        required_parameters = [
            "item_embedding_dim",
            "cate_embedding_dim",
            "max_seq_length",
            "loss",
            "method",
            "user_vocab",
            "item_vocab",
            "cate_vocab",
            "attention_size",
            "hidden_size",
            "att_fcn_layer_sizes",
            "discrepancy_loss_weight",
            "contrastive_loss_weight",
            "is_clip_norm",
            "contrastive_length_threshold",
        ]
    elif f_config["model_type"] in [
        "nextitnet",
        "next_it_net",
        "NextItNet",
        "NEXT_IT_NET",
    ]:
        required_parameters = [
            "item_embedding_dim",
            "cate_embedding_dim",
            "user_embedding_dim",
            "max_seq_length",
            "loss",
            "method",
            "user_vocab",
            "item_vocab",
            "cate_vocab",
            "dilations",
            "kernel_size",
            "min_seq_length",
        ]
    else:
        required_parameters = []

    # check required parameters
    for param in required_parameters:
        if param not in f_config:
            raise ValueError("Parameters {0} must be set".format(param))

    if f_config["model_type"] in ["exDeepFM", "xDeepFM"]:
        if f_config["data_format"] != "ffm":
            raise ValueError(
                "For xDeepFM model, data format must be 'ffm', but your set is {0}".format(
                    f_config["data_format"]
                )
            )
    elif f_config["model_type"] in ["dkn", "DKN"]:
        if f_config["data_format"] != "dkn":
            raise ValueError(
                "For dkn model, data format must be 'dkn', but your set is {0}".format(
                    f_config["data_format"]
                )
            )
    check_type(f_config)


def load_yaml(filename):
    """Load a yaml file.

    Args:
        filename (str): Filename.

    Returns:
        dict: Dictionary.
    """
    try:
        with open(filename, "r") as f:
            config = yaml.load(f, yaml.SafeLoader)
        return config
    except FileNotFoundError:  # for file not found
        raise
    except Exception as e:  # for other exceptions
        raise IOError("load {0} error!".format(filename))


def create_hparams(flags):
    """Create the model hyperparameters.

    Args:
        flags (dict): Dictionary with the model requirements.

    Returns:
        obj: Hyperparameter object in TF (tf.contrib.training.HParams).
    """
    return tf.contrib.training.HParams(
        # data
        kg_file=flags["kg_file"] if "kg_file" in flags else None,
        user_clicks=flags["user_clicks"] if "user_clicks" in flags else None,
        FEATURE_COUNT=flags["FEATURE_COUNT"] if "FEATURE_COUNT" in flags else None,
        FIELD_COUNT=flags["FIELD_COUNT"] if "FIELD_COUNT" in flags else None,
        data_format=flags["data_format"] if "data_format" in flags else None,
        PAIR_NUM=flags["PAIR_NUM"] if "PAIR_NUM" in flags else None,
        DNN_FIELD_NUM=flags["DNN_FIELD_NUM"] if "DNN_FIELD_NUM" in flags else None,
        n_user=flags["n_user"] if "n_user" in flags else None,
        n_item=flags["n_item"] if "n_item" in flags else None,
        n_user_attr=flags["n_user_attr"] if "n_user_attr" in flags else None,
        n_item_attr=flags["n_item_attr"] if "n_item_attr" in flags else None,
        iterator_type=flags["iterator_type"] if "iterator_type" in flags else None,
        SUMMARIES_DIR=flags["SUMMARIES_DIR"] if "SUMMARIES_DIR" in flags else None,
        MODEL_DIR=flags["MODEL_DIR"] if "MODEL_DIR" in flags else None,
        # dkn
        wordEmb_file=flags["wordEmb_file"] if "wordEmb_file" in flags else None,
        entityEmb_file=flags["entityEmb_file"] if "entityEmb_file" in flags else None,
        contextEmb_file=flags["contextEmb_file"]
        if "contextEmb_file" in flags
        else None,
        news_feature_file=flags["news_feature_file"]
        if "news_feature_file" in flags
        else None,
        user_history_file=flags["user_history_file"]
        if "user_history_file" in flags
        else None,
        use_entity=flags["use_entity"] if "use_entity" in flags else True,
        use_context=flags["use_context"] if "use_context" in flags else True,
        doc_size=flags["doc_size"] if "doc_size" in flags else None,
        history_size=flags["history_size"] if "history_size" in flags else None,
        word_size=flags["word_size"] if "word_size" in flags else None,
        entity_size=flags["entity_size"] if "entity_size" in flags else None,
        entity_dim=flags["entity_dim"] if "entity_dim" in flags else None,
        entity_embedding_method=flags["entity_embedding_method"]
        if "entity_embedding_method" in flags
        else None,
        transform=flags["transform"] if "transform" in flags else None,
        train_ratio=flags["train_ratio"] if "train_ratio" in flags else None,
        # model
        dim=flags["dim"] if "dim" in flags else None,
        layer_sizes=flags["layer_sizes"] if "layer_sizes" in flags else None,
        cross_layer_sizes=flags["cross_layer_sizes"]
        if "cross_layer_sizes" in flags
        else None,
        cross_layers=flags["cross_layers"] if "cross_layers" in flags else None,
        activation=flags["activation"] if "activation" in flags else None,
        cross_activation=flags["cross_activation"]
        if "cross_activation" in flags
        else "identity",
        user_dropout=flags["user_dropout"] if "user_dropout" in flags else False,
        dropout=flags["dropout"] if "dropout" in flags else [0.0],
        attention_layer_sizes=flags["attention_layer_sizes"]
        if "attention_layer_sizes" in flags
        else None,
        attention_activation=flags["attention_activation"]
        if "attention_activation" in flags
        else None,
        attention_dropout=flags["attention_dropout"]
        if "attention_dropout" in flags
        else 0.0,
        model_type=flags["model_type"] if "model_type" in flags else None,
        method=flags["method"] if "method" in flags else None,
        load_saved_model=flags["load_saved_model"]
        if "load_saved_model" in flags
        else False,
        load_model_name=flags["load_model_name"]
        if "load_model_name" in flags
        else None,
        filter_sizes=flags["filter_sizes"] if "filter_sizes" in flags else None,
        num_filters=flags["num_filters"] if "num_filters" in flags else None,
        mu=flags["mu"] if "mu" in flags else None,
        fast_CIN_d=flags["fast_CIN_d"] if "fast_CIN_d" in flags else 0,
        use_Linear_part=flags["use_Linear_part"]
        if "use_Linear_part" in flags
        else False,
        use_FM_part=flags["use_FM_part"] if "use_FM_part" in flags else False,
        use_CIN_part=flags["use_CIN_part"] if "use_CIN_part" in flags else False,
        use_DNN_part=flags["use_DNN_part"] if "use_DNN_part" in flags else False,
        # train
        init_method=flags["init_method"] if "init_method" in flags else "tnormal",
        init_value=flags["init_value"] if "init_value" in flags else 0.01,
        embed_l2=flags["embed_l2"] if "embed_l2" in flags else 0.0000,
        embed_l1=flags["embed_l1"] if "embed_l1" in flags else 0.0000,
        layer_l2=flags["layer_l2"] if "layer_l2" in flags else 0.0000,
        layer_l1=flags["layer_l1"] if "layer_l1" in flags else 0.0000,
        cross_l2=flags["cross_l2"] if "cross_l2" in flags else 0.0000,
        cross_l1=flags["cross_l1"] if "cross_l1" in flags else 0.0000,
        attn_loss_weight=flags["attn_loss_weight"] if "attn_loss_weight" in flags else 0.0000,
        contrastive_loss=flags["contrastive_loss"] if "contrastive_loss" in flags else "bpr",
        triplet_margin=flags["triplet_margin"] if "triplet_margin" in flags else 1.0,
        discrepancy_loss_weight=flags["discrepancy_loss_weight"] if "discrepancy_loss_weight" in flags else 0.0000,
        contrastive_loss_weight=flags["contrastive_loss_weight"] if "contrastive_loss_weight" in flags else 0.0000,
        contrastive_length_threshold=flags["contrastive_length_threshold"] if "contrastive_length_threshold" in flags else 1,
        contrastive_recent_k=flags["contrastive_recent_k"] if "contrastive_recent_k" in flags else 3,
        reg_kg=flags["reg_kg"] if "reg_kg" in flags else 0.0000,
        learning_rate=flags["learning_rate"] if "learning_rate" in flags else 0.001,
        lr_rs=flags["lr_rs"] if "lr_rs" in flags else 1,
        lr_kg=flags["lr_kg"] if "lr_kg" in flags else 0.5,
        kg_training_interval=flags["kg_training_interval"]
        if "kg_training_interval" in flags
        else 5,
        max_grad_norm=flags["max_grad_norm"] if "max_grad_norm" in flags else 2,
        is_clip_norm=flags["is_clip_norm"] if "is_clip_norm" in flags else 0,
        vector_alpha=flags["vector_alpha"] if "vector_alpha" in flags else False,
        manual_alpha=flags["manual_alpha"] if "manual_alpha" in flags else False,
        manual_alpha_value=flags["manual_alpha_value"] if "manual_alpha_value" in flags else 0.5,
        interest_evolve=flags["interest_evolve"] if "interest_evolve" in flags else True,
        predict_long_short=flags["predict_long_short"] if "predict_long_short" in flags else True,
        dtype=flags["dtype"] if "dtype" in flags else 32,
        loss=flags["loss"] if "loss" in flags else None,
        optimizer=flags["optimizer"] if "optimizer" in flags else "adam",
        epochs=flags["epochs"] if "epochs" in flags else 10,
        batch_size=flags["batch_size"] if "batch_size" in flags else 1,
        enable_BN=flags["enable_BN"] if "enable_BN" in flags else False,
        # show info
        show_step=flags["show_step"] if "show_step" in flags else 1,
        save_model=flags["save_model"] if "save_model" in flags else True,
        save_epoch=flags["save_epoch"] if "save_epoch" in flags else 5,
        metrics=flags["metrics"] if "metrics" in flags else None,
        write_tfevents=flags["write_tfevents"] if "write_tfevents" in flags else False,
        # sequential
        item_embedding_dim=flags["item_embedding_dim"]
        if "item_embedding_dim" in flags
        else None,
        cate_embedding_dim=flags["cate_embedding_dim"]
        if "cate_embedding_dim" in flags
        else None,
        user_embedding_dim=flags["user_embedding_dim"]
        if "user_embedding_dim" in flags
        else None,
        train_num_ngs=flags["train_num_ngs"] if "train_num_ngs" in flags else 4,
        need_sample=flags["need_sample"] if "need_sample" in flags else True,
        embedding_dropout=flags["embedding_dropout"]
        if "embedding_dropout" in flags
        else 0.3,
        user_vocab=flags["user_vocab"] if "user_vocab" in flags else None,
        item_vocab=flags["item_vocab"] if "item_vocab" in flags else None,
        cate_vocab=flags["cate_vocab"] if "cate_vocab" in flags else None,
        pairwise_metrics=flags["pairwise_metrics"]
        if "pairwise_metrics" in flags
        else None,
        weighted_metrics=flags["weighted_metrics"]
        if "weighted_metrics" in flags
        else None,
        EARLY_STOP=flags["EARLY_STOP"] if "EARLY_STOP" in flags else 100,
        # gru4rec
        max_seq_length=flags["max_seq_length"] if "max_seq_length" in flags else None,
        hidden_size=flags["hidden_size"] if "hidden_size" in flags else None,
        # caser,
        L=flags["L"] if "L" in flags else None,
        T=flags["T"] if "T" in flags else None,
        n_v=flags["n_v"] if "n_v" in flags else None,
        n_h=flags["n_h"] if "n_h" in flags else None,
        min_seq_length=flags["min_seq_length"] if "min_seq_length" in flags else 1,
        # sli_rec
        attention_size=flags["attention_size"] if "attention_size" in flags else None,
        att_fcn_layer_sizes=flags["att_fcn_layer_sizes"]
        if "att_fcn_layer_sizes" in flags
        else None,
        # nextitnet
        dilations=flags["dilations"] if "dilations" in flags else None,
        kernel_size=flags["kernel_size"] if "kernel_size" in flags else None,
        # lightgcn
        embed_size=flags["embed_size"] if "embed_size" in flags else None,
        n_layers=flags["n_layers"] if "n_layers" in flags else None,
        decay=flags["decay"] if "decay" in flags else None,
        eval_epoch=flags["eval_epoch"] if "eval_epoch" in flags else None,
        top_k=flags["top_k"] if "top_k" in flags else None,
        counterfactual_recent_k=flags["counterfactual_recent_k"] if "counterfactual_recent_k" in flags else 5,
        use_complex_attention=flags["use_complex_attention"] if "use_complex_attention" in flags else False,
        sequential_model=flags["sequential_model"] if "sequential_model" in flags else 'time4lstm',
        time_unit=flags["time_unit"] if "time_unit" in flags else 's',
        ncf_layer_sizes=flags["ncf_layer_sizes"] if "ncf_layer_sizes" in flags else [80, 40],
    )


def prepare_hparams(yaml_file=None, **kwargs):
    """Prepare the model hyperparameters and check that all have the correct value.

    Args:
        yaml_file (str): YAML file as configuration.

    Returns:
        obj: Hyperparameter object in TF (tf.contrib.training.HParams).
    """
    if yaml_file is not None:
        config = load_yaml(yaml_file)
        config = flat_config(config)
    else:
        config = {}

    if kwargs:
        for name, value in six.iteritems(kwargs):
            config[name] = value

    check_nn_config(config)
    return create_hparams(config)


def download_deeprec_resources(azure_container_url, data_path, remote_resource_name):
    """Download resources.

    Args:
        azure_container_url (str): URL of Azure container.
        data_path (str): Path to download the resources.
        remote_resource_name (str): Name of the resource.
    """
    os.makedirs(data_path, exist_ok=True)
    remote_path = azure_container_url + remote_resource_name
    maybe_download(remote_path, remote_resource_name, data_path)
    zip_ref = zipfile.ZipFile(os.path.join(data_path, remote_resource_name), "r")
    zip_ref.extractall(data_path)
    zip_ref.close()
    os.remove(os.path.join(data_path, remote_resource_name))


def mrr_score(y_true, y_score):
    """Computing mrr score metric.

    Args:
        y_true (np.ndarray): ground-truth labels.
        y_score (np.ndarray): predicted labels.
    
    Returns:
        np.ndarray: mrr scores.
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def ndcg_score(y_true, y_score, k=10):
    """Computing ndcg score metric at k.

    Args:
        y_true (np.ndarray): ground-truth labels.
        y_score (np.ndarray): predicted labels.

    Returns:
        np.ndarray: ndcg scores.
    """
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def hit_score(y_true, y_score, k=10):
    """Computing hit score metric at k.

    Args:
        y_true (np.ndarray): ground-truth labels.
        y_score (np.ndarray): predicted labels.

    Returns:
        np.ndarray: hit score.
    """
    ground_truth = np.where(y_true == 1)[0]
    argsort = np.argsort(y_score)[::-1][:k]
    for idx in argsort:
        if idx in ground_truth:
            return 1
    return 0


def dcg_score(y_true, y_score, k=10):
    """Computing dcg score metric at k.

    Args:
        y_true (np.ndarray): ground-truth labels.
        y_score (np.ndarray): predicted labels.

    Returns:
        np.ndarray: dcg scores.
    """
    k = min(np.shape(y_true)[-1], k)
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def cal_metric(labels, preds, metrics):
    """Calculate metrics,such as auc, logloss.
    
    FIXME: 
        refactor this with the reco metrics and make it explicit.
    """
    res = {}
    if not metrics:
        return res

    for metric in metrics:
        if metric == "auc":
            auc = roc_auc_score(np.asarray(labels), np.asarray(preds))
            res["auc"] = round(auc, 4)
        elif metric == "rmse":
            rmse = mean_squared_error(np.asarray(labels), np.asarray(preds))
            res["rmse"] = np.sqrt(round(rmse, 4))
        elif metric == "logloss":
            # avoid logloss nan
            preds = [max(min(p, 1.0 - 10e-12), 10e-12) for p in preds]
            logloss = log_loss(np.asarray(labels), np.asarray(preds))
            res["logloss"] = round(logloss, 4)
        elif metric == "acc":
            pred = np.asarray(preds)
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            acc = accuracy_score(np.asarray(labels), pred)
            res["acc"] = round(acc, 4)
        elif metric == "f1":
            pred = np.asarray(preds)
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            f1 = f1_score(np.asarray(labels), pred)
            res["f1"] = round(f1, 4)
        elif metric == "mean_mrr":
            mean_mrr = np.mean(
                [
                    mrr_score(each_labels, each_preds)
                    for each_labels, each_preds in zip(labels, preds)
                ]
            )
            res["mean_mrr"] = round(mean_mrr, 4)
        elif metric.startswith("ndcg"):  # format like:  ndcg@2;4;6;8
            ndcg_list = [1, 2]
            ks = metric.split("@")
            if len(ks) > 1:
                ndcg_list = [int(token) for token in ks[1].split(";")]
            for k in ndcg_list:
                ndcg_temp = np.mean(
                    [
                        ndcg_score(each_labels, each_preds, k)
                        for each_labels, each_preds in zip(labels, preds)
                    ]
                )
                res["ndcg@{0}".format(k)] = round(ndcg_temp, 4)
        elif metric.startswith("hit"):  # format like:  hit@2;4;6;8
            hit_list = [1, 2]
            ks = metric.split("@")
            if len(ks) > 1:
                hit_list = [int(token) for token in ks[1].split(";")]
            for k in hit_list:
                hit_temp = np.mean(
                    [
                        hit_score(each_labels, each_preds, k)
                        for each_labels, each_preds in zip(labels, preds)
                    ]
                )
                res["hit@{0}".format(k)] = round(hit_temp, 4)
        elif metric == "group_auc":
            group_auc = np.mean(
                [
                    roc_auc_score(each_labels, each_preds)
                    for each_labels, each_preds in zip(labels, preds)
                ]
            )
            res["group_auc"] = round(group_auc, 4)
        else:
            raise ValueError("not define this metric {0}".format(metric))
    return res


def cal_weighted_metric(users, preds, labels, metrics):

    res = {}
    if not metrics:
        return res

    df = pd.DataFrame({'users': users, 'preds': preds, 'labels': labels})
    weight = df[["users", "labels"]].groupby("users").count().reset_index().set_index("users", drop=True).rename(columns={"labels": "weight"})
    weight["weight"] = weight["weight"]/weight["weight"].sum()
    for metric in metrics:
        if metric == 'wauc':
            wauc = cal_wauc(df, weight)
            res["wauc"] = round(wauc, 4)
        elif metric == 'wmrr':
            wmrr = cal_wmrr(df, weight)
            res["wmrr"] = round(wmrr, 4)
        elif metric.startswith("whit"): # format like: whit@2;4;6;8
            hit_list = [1, 2]
            ks = metric.split("@")
            if len(ks) > 1:
                hit_list = [int(token) for token in ks[1].split(";")]
            whit_res = cal_whit(df, weight, hit_list)
            res.update(whit_res)
        elif metric.startswith("wndcg"): # format like: wndcg@2;4;6;8
            ndcg_list = [1, 2]
            ks = metric.split("@")
            if len(ks) > 1:
                ndcg_list = [int(token) for token in ks[1].split(";")]
            wndcg_res = cal_wndcg(df, weight, ndcg_list)
            res.update(wndcg_res)
        else:
            raise ValueError("not define this metric {0}".format(metric))
    return res


def cal_wauc(df, weight):

    weight["auc"] = df.groupby("users").apply(groupby_auc)
    wauc_score = (weight["weight"]*weight["auc"]).sum()
    weight.drop(columns="auc", inplace=True)

    return wauc_score


def groupby_auc(df):

    y_hat = df.preds
    y = df.labels
    return roc_auc_score(y, y_hat)


def cal_wmrr(df, weight):

    weight["mrr"] = df.groupby("users").apply(groupby_mrr)
    wmrr_score = (weight["weight"]*weight["mrr"]).sum()
    weight.drop(columns="mrr", inplace=True)

    return wmrr_score


def groupby_mrr(df):

    y_hat = df.preds
    y = df.labels
    return mrr_score(y, y_hat)


def cal_whit(df, weight, hit_list):

    whit_res = {}
    weight["hit"] = df.groupby("users").apply(groupby_hit, hit_list=hit_list)
    whit_score = (weight["weight"]*weight["hit"]).sum()
    weight.drop(columns="hit", inplace=True)
    for i, k in enumerate(hit_list):
        metric = "whit@{0}".format(k)
        whit_res[metric] = round(whit_score[i], 4)

    return whit_res


def groupby_hit(df, hit_list):

    y_hat = df.preds
    y = df.labels
    hit = np.array([hit_score(y, y_hat, k) for k in hit_list])

    return hit


def cal_wndcg(df, weight, ndcg_list):

    wndcg_res = {}
    weight["ndcg"] = df.groupby("users").apply(groupby_ndcg, ndcg_list=ndcg_list)
    wndcg_score = (weight["weight"]*weight["ndcg"]).sum()
    weight.drop(columns="ndcg", inplace=True)
    for i, k in enumerate(ndcg_list):
        metric = "wndcg@{0}".format(k)
        wndcg_res[metric] = round(wndcg_score[i], 4)

    return wndcg_res


def groupby_ndcg(df, ndcg_list):

    y_hat = df.preds
    y = df.labels
    ndcg = np.array([ndcg_score(y, y_hat, k) for k in ndcg_list])

    return ndcg


def cal_mean_alpha_metric(alphas, labels):

    res = {}

    alphas = np.asarray(alphas)
    labels = np.asarray(labels)
    res["mean_alpha"] = round((alphas*labels).sum()/labels.sum(), 4)

    return res


def load_dict(filename):
    """Load the vocabularies.

    Args:
        filename (str): Filename of user, item or category vocabulary.

    Returns:
        dict: A saved vocabulary.
    """
    with open(filename, "rb") as f:
        f_pkl = pkl.load(f)
        return f_pkl


def dice(_x, axis=-1, epsilon=0.000000001, name=''):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        alphas = tf.get_variable('alpha'+name, _x.get_shape()[-1],
                                initializer=tf.constant_initializer(0.0),
                                dtype=tf.float32)
        input_shape = list(_x.get_shape())

        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[axis] = input_shape[axis]

    # case: train mode (uses stats of the current batch)
    mean = tf.reduce_mean(_x, axis=reduction_axes)
    brodcast_mean = tf.reshape(mean, broadcast_shape)
    std = tf.reduce_mean(tf.square(_x - brodcast_mean) +
                        epsilon, axis=reduction_axes)
    std = tf.math.sqrt(std)
    brodcast_std = tf.reshape(std, broadcast_shape)
    x_normed = (_x - brodcast_mean) / (brodcast_std + epsilon)
    # x_normed = tf.layers.batch_normalization(_x, center=False, scale=False)
    x_p = tf.sigmoid(x_normed)

    return alphas * (1.0 - x_p) * _x + x_p * _x
