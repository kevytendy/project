from data import NumericalField, CategoricalField, Iterator
from data import Dataset
from synthesizer import VGAN_generator, VGAN_discriminator
from synthesizer import LGAN_generator, LGAN_discriminator, LSTM_discriminator
from synthesizer import DCGAN_generator, DCGAN_discriminator
from synthesizer import V_Train, C_Train, W_Train, C_Train_dp, C_Train_nofair
from random import choice
import multiprocessing
import pandas as pd
import numpy as np
import torch
import argparse
import json
import os

# Variable definitions for models
VGAN_variable = {
    "batch_size": [128, 512, 256],
    "z_dim": [128, 256],
    "gen_hidden_dim": [100, 200, 300, 400],
    "gen_num_layers": [1, 2, 3],
    "dis_hidden_dim": [100, 200, 300, 400],
    "dis_num_layers": [1, 2, 3],
    "dis_lstm_dim": [100, 200, 300, 400],
    "lr": [0.0001, 0.0002, 0.001, 0.0005],
    "noise": [0.05, 0.1, 0.2, 0.3]
}

LGAN_variable = {
    "batch_size": [50, 100, 200],
    "z_dim": [50, 100, 200, 400],
    "gen_feature_dim": [100, 200, 300, 400, 500, 600],
    "gen_lstm_dim": [100, 200, 300, 400, 500, 600],
    "dis_hidden_dim": [100, 200, 300, 400, 500],
    "dis_num_layers": [1, 2, 3, 4, 5],
    "dis_lstm_dim": [100, 200, 300, 400],
    "lr": [0.0002, 0.0001, 0.0005, 0.001],
    "noise": [0.05, 0.1, 0.2, 0.3]
}

DCGAN_variable = {
    "batch_size": [50, 100, 150],
    "z_dim": [50, 100, 200, 300, 400],
    "lr": [0.0005, 0.0001, 0.0002, 0.0003]
}

def parameter_search(Model):
    parameters = {}
    if Model == "VGAN":
        variable = VGAN_variable
    elif Model == "LGAN":
        variable = LGAN_variable
    elif Model == "DCGAN":
        variable = DCGAN_variable
    for param in variable.keys():
        parameters[param] = choice(variable[param])
    return parameters

def thread_run(path, search, config, col_type, dataset, sampleset):
    if config.get("rand_search", "no") == "yes":
        param = parameter_search(config["model"])
    else:
        param = config["param"]

    with open(os.path.join(path, "exp_params.json"), "a") as f:
        json.dump(param, f)
        f.write("\n")

    model = config["model"]
    train_method = config["train_method"]

    labels = config.get("label") if train_method in ["CTrain", "CTrain_dp", "CTrain_nofair"] else None
    square = model == "DCGAN"
    pad = 0 if square else None

    train_it, sample_it = Iterator.split(
        batch_size=param["batch_size"],
        train=dataset,
        validation=sampleset,
        sort_key=None,
        shuffle=True,
        labels=labels,
        square=square,
        pad=pad
    )
    x_dim = train_it.data.shape[1]
    col_dim = dataset.col_dim
    col_ind = dataset.col_ind

    c_dim = train_it.label.shape[1] if train_method in ["CTrain", "CTrain_dp", "CTrain_nofair"] else 0
    condition = c_dim > 0

    if model == "VGAN":
        gen = VGAN_generator(param["z_dim"], param["gen_hidden_dim"], x_dim, param["gen_num_layers"], col_type, col_ind,
                             condition=condition, c_dim=c_dim)
        dis_model = config.get("dis_model", "mlp")
        if dis_model == "lstm":
            dis = LSTM_discriminator(x_dim, param["dis_lstm_dim"], condition, c_dim)
        else:
            dis = VGAN_discriminator(x_dim, param["dis_hidden_dim"], param["dis_num_layers"], condition, c_dim)
    elif model == "LGAN":
        gen = LGAN_generator(param["z_dim"], param["gen_feature_dim"], param["gen_lstm_dim"], col_dim, col_type,
                             condition, c_dim)
        dis_model = config.get("dis_model", "mlp")
        if dis_model == "lstm":
            dis = LSTM_discriminator(x_dim, param["dis_lstm_dim"], condition, c_dim)
        else:
            dis = LGAN_discriminator(x_dim, param["dis_hidden_dim"], param["dis_num_layers"], condition, c_dim)

    GPU = torch.cuda.is_available()

    sample_times = config.get("sample_times", 1)

    if train_method == "VTrain":
        KL = config.get("KL", "yes") == "yes"
        V_Train(search, path, sample_it, gen, dis, config["n_epochs"], param["lr"], train_it, param["z_dim"], dataset,
                col_type, sample_times, itertimes=100, steps_per_epoch=config["steps_per_epoch"], GPU=GPU, KL=KL)
    elif train_method == "CTrain":
        C_Train(search, path, sample_it, gen, dis, config["n_epochs"], param["lr"], train_it, param["z_dim"], dataset,
                col_type, sample_times, itertimes=100, steps_per_epoch=config["steps_per_epoch"], GPU=GPU)
    elif train_method == "WTrain":
        dis.wgan = True
        KL = config.get("KL", "yes") == "yes"
        W_Train(search, path, sample_it, gen, dis, config["ng"], config["nd"], 0.01, param["lr"], train_it,
                param["z_dim"], dataset, col_type, sample_times, itertimes=100, GPU=GPU, KL=KL)
    elif train_method == "CTrain_dp":
        dis.wgan = True
        C_Train_dp(search, path, sample_it, gen, dis, config["ng"], config["nd"], 0.01, param["lr"], train_it,
                   param["z_dim"], dataset, col_type, config["eps"], sample_times, itertimes=100, GPU=GPU)
    elif train_method == "CTrain_nofair":
        C_Train_nofair(search, path, sample_it, gen, dis, config["n_epochs"], param["lr"], train_it, param["z_dim"],
                       dataset, col_type, sample_times, itertimes=100, steps_per_epoch=config["steps_per_epoch"],
                       GPU=GPU)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    parser = argparse.ArgumentParser()
    parser.add_argument('configs', help='a json config file')
    args = parser.parse_args()
    with open(args.configs) as f:
        configs = json.load(f)

    os.makedirs("expdir", exist_ok=True)

    for config in configs:
        path = os.path.join("expdir", config["name"])
        os.makedirs(path, exist_ok=True)

        train = pd.read_csv(config["train"])
        fields = []
        col_type = []

        ratio = config.get("ratio", 1)
        cond = config.get("label", [])
        noise = choice([0.05, 0.1, 0.2, 0.3])

        for i, col in enumerate(train.columns):
            if col in cond:
                fields.append((col, CategoricalField("one-hot", noise=0)))
                col_type.append("condition")
            elif i in config["normalize_cols"]:
                fields.append((col, NumericalField("normalize")))
                col_type.append("normalize")
            elif i in config["gmm_cols"]:
                fields.append((col, NumericalField("gmm", n=5)))
                col_type.append("gmm")
            elif i in config["one-hot_cols"]:
                fields.append((col, CategoricalField("one-hot", noise=noise)))
                col_type.append("one-hot")
            elif i in config["ordinal_cols"]:
                fields.append((col, CategoricalField("dict")))
                col_type.append("ordinal")
            else:
                fields.append((col, CategoricalField("binary", noise=noise)))
                col_type.append("binary")

        trn, samp = Dataset.split(
            fields=fields,
            path=".",
            train=config["train"],
            validation=config["sample"],
            format="csv",
            valid_ratio=ratio
        )
        trn.learn_convert()
        samp.learn_convert()

        print(f"train row : {len(trn)}")
        print(f"sample row: {len(samp)}")

        n_search = config["n_search"]
        jobs = [multiprocessing.Process(target=thread_run, args=(path, search, config, col_type, trn, samp)) for search
                in range(n_search)]
        for j in jobs:
            j.start()
        for j in jobs:
            j.join()