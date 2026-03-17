"""
This file is used in conjunction with the shared data prep file. This file extends shared data prep's capabilities to function on lists.
It also loads the data for usage with the neural networks with the aid of shared data prep. It extracts features (removing prompts while doing so),
and remapping everything to integers to be used in tensors. Finally it embeds the essays and stores this data so it isn't reprocessed every time.

Written By: Alexander Lenz
"""

import torch
import pickle
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sentence_transformers import SentenceTransformer
from shared_data_prep import load_all_data, denormalize_asap1, denormalize_asap2

#simple code to create dataloaders from text and truth values
def create_loader(X_test, Y_test):
    test_dataset = TensorDataset(X_test, Y_test)

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    return test_loader

#this works in conjunction with the denormalize scores to make it work on functions
def denormalize_all_scores(y_pred_list_train, y_true_list_train, y_pred_list_val, y_true_list_val, y_pred_list_test, y_true_list_test):
    y_pred_list_train_denorm = []
    y_true_list_train_denorm = []
    y_pred_list_val_denorm = []
    y_true_list_val_denorm = []
    y_pred_list_test_denorm = []
    y_true_list_test_denorm = []

    length = 0

    if y_pred_list_train is not None:
        length = len(y_pred_list_train)
    elif y_pred_list_val is not None:
        length = len(y_pred_list_val)
    elif y_pred_list_test is not None:
        length = len(y_pred_list_test)
    else:
        raise Exception("Can't Normalize Empty Lists")

    for i in range(length):
        if y_pred_list_train is not None and y_true_list_train is not None:
            pred, true = denormalize_scores(y_pred_list_train[i], y_true_list_train[i], 1)
            y_pred_list_train_denorm.append(pred)
            y_true_list_train_denorm.append(true)

        if y_pred_list_val is not None and y_true_list_val is not None:
            pred, true = denormalize_scores(y_pred_list_val[i], y_true_list_val[i], 1)
            y_pred_list_val_denorm.append(pred)
            y_true_list_val_denorm.append(true)

        if y_pred_list_test is not None and y_true_list_test is not None:
            pred, true = denormalize_scores(y_pred_list_test[i], y_true_list_test[i], 2)
            y_pred_list_test_denorm.append(pred)
            y_true_list_test_denorm.append(true)

    return y_pred_list_train_denorm, y_true_list_train_denorm, y_pred_list_val_denorm, y_true_list_val_denorm, y_pred_list_test_denorm, y_true_list_test_denorm

#this function distinguishes between asap1 and 2 to call the correct functions
def denormalize_scores(y_pred_list, y_true_list, asap_version):
    y_pred_list_denorm = []
    y_true_list_denorm = []

    y_pred_list_copy = y_pred_list.copy()
    y_true_list_copy = y_true_list.copy()

    for i in range(len(y_pred_list)):
        prompt_id = y_true_list[i][1]

        if asap_version == 1:
            y_pred_list_copy[i] = int(round(denormalize_asap1(y_pred_list_copy[i], prompt_id)))
            y_pred_list_denorm.append(y_pred_list_copy[i])

            temp = y_true_list_copy[i].copy()

            temp[0] = int(round(denormalize_asap1(temp[0], prompt_id)))
            y_true_list_denorm.append(temp)
        else:
            y_pred_list_copy[i] = int(round(denormalize_asap2(y_pred_list_copy[i])))
            y_pred_list_denorm.append(y_pred_list_copy[i])

            temp = y_true_list_copy[i].copy()

            temp[0] = int(round(denormalize_asap2(temp[0])))
            y_true_list_denorm.append(temp)

    return y_pred_list_denorm, y_true_list_denorm

#uses navids code to load data and store it
def load_everything_navid():
    #nltk.download('punkt_tab')

    return embed_and_pickle()

#goes through all the demographic information converting to integers to make it usable in tensors
def convert_asap2_demographics(asap2):
    print("Convert demographics")
    print("Initial shape: " + str(asap2.shape))

    asap2.loc[:, "economically_disadvantaged"] = asap2["economically_disadvantaged"].map({"Economically disadvantaged": 1, "Not economically disadvantaged": 0})
    asap2.loc[:, "student_disability_status"] = asap2["student_disability_status"].map({"Identified as having disability": 1, "Not identified as having disability": 0})
    asap2.loc[:, "ell_status"] = asap2["ell_status"].map({"Yes": 1, "No": 0})
    asap2.loc[:, "gender"] = asap2["gender"].map({"M": 1, "F": 0})
    asap2.loc[:, "race_ethnicity"] = asap2["race_ethnicity"].map({"American Indian/Alaskan Native": 0, "Asian/Pacific Islander": 1, "Black/African American": 2, "Hispanic/Latino": 3, "Two or more races/Other": 4, "White": 5})

    asap2 = asap2.fillna(value = -1)

    print("Final shape: " + str(asap2.shape))

    return asap2

#uses navids code to load data, embed it,and store it
def embed_and_pickle():
    try:
        data = pickle.load(open('NeuralNetworks/alex_embeddings_no_prompt_with_demographics.pkl', 'rb'))
        print("found file")

        train = data["asap1_train"]
        val = data["asap1_val"]
        test = data["asap2"]

        X_train = train["X_train"]
        Y_train = train["Y_train"]
        X_val = val["X_val"]
        Y_val = val["Y_val"]
        X_test = test["X_test"]
        Y_test = test["Y_test"]



        return X_train, Y_train, X_val, Y_val, X_test, Y_test
    except FileNotFoundError:
        print("file not found")

    data = load_all_data()

    train = data["asap1_train"].dropna()
    val = data["asap1_val"]
    test = data["asap2"]

    test = convert_asap2_demographics(test)

    print("removing prompts from essays")

    train.loc[:, "text"] = remove_prompt_from_essays(train["text"])
    val.loc[:, "text"] = remove_prompt_from_essays(val["text"])
    test.loc[:, "text"] = remove_prompt_from_essays(test["text"])

    print("prompts removed")

    model = SentenceTransformer('all-mpnet-base-v2')

    print("embed train")
    X_train = torch.tensor(np.array(list(map(model.encode, train["text"]))), dtype=torch.float32)
    Y_train = torch.tensor(train[["norm_score", "prompt_id"]].values, dtype=torch.float32)
    print("embed val")
    X_val = torch.tensor(np.array(list(map(model.encode, val["text"]))), dtype=torch.float32)
    Y_val = torch.tensor(val[["norm_score", "prompt_id"]].values, dtype=torch.float32)
    print("embed test")
    X_test = torch.tensor(np.array(list(map(model.encode, test["text"]))), dtype=torch.float32)
    test = np.array(test[["norm_score", "prompt_id", "economically_disadvantaged", "student_disability_status", "ell_status", "race_ethnicity", "gender"]], dtype=np.float32)
    Y_test = torch.tensor(test, dtype=torch.float32)

    print("embed train")

    train = {"X_train": X_train, "Y_train": Y_train}
    val = {"X_val": X_val, "Y_val": Y_val}
    test = {"X_test": X_test, "Y_test": Y_test}

    data = {'asap1_train': train, 'asap1_val': val, 'asap2': test}

    with open('NeuralNetworks/alex_embeddings_no_prompt_with_demographics.pkl', 'wb') as f:
        pickle.dump(data, f)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

#removes the prompt (first sentence) from the essays
def remove_prompt_from_essays(essays):
    new_essays = []
    for essay in essays:
        new_essays.append(essay[essay.find("[SEP]") + 6:])
    return new_essays