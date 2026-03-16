"""
This file pulls from all the other files to train, test, and visualize the results of the neural networks. It also contains code to run inferences which was created for the notebook.

Written By: Alexander Lenz
"""

import MLP
import BNN
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import cohen_kappa_score
import pickle
from Data_processing import create_loader, denormalize_scores, load_everything_navid
from Analysis import run_and_plot_nns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_up_train_test_plot():
    print("device is " + str(torch.device))

    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_everything_navid()

    print(X_train.shape)
    print(Y_train.shape)
    print(X_val.shape)
    print(Y_val.shape)
    print(X_test.shape)
    print(Y_test.shape)

    train_loader = create_loader(X_train, Y_train)
    val_loader = create_loader(X_val, Y_val)
    test_loader = create_loader(X_test, Y_test)


    cells = [768, int(768/1.5), int(768/1.5/1.5)]
    #cells = [300, 300, 300]

    mlp = MLP.MLP(in_features=X_train.shape[1], out_features=1, cells=cells).to(device)

    bnn = BNN.BNN(in_features=X_train.shape[1], out_features=1, cells=cells).to(device)

    print("MLP model: " + str(mlp.model))
    print("BNN model: " + str(bnn.model))

    nns = [mlp, bnn]

    run_and_plot_nns(nns, train_loader, val_loader, test_loader)

    torch.save(mlp, "NeuralNetworks/MLP_model")
    torch.save(bnn, "NeuralNetworks/BNN_model")

def inference_data_visualization(nn_name, y_pred_list_val, y_true_list_val, y_pred_list_test, y_true_list_test, plot):

    y_pred_val, y_true_val = denormalize_scores(y_pred_list_val, y_true_list_val, 1)
    y_true_val_true_only = np.array(y_true_val)[:, 0]

    val_qwk = (cohen_kappa_score(y_pred_val, y_true_val_true_only, weights='quadratic'))

    y_pred_test, y_true_test = denormalize_scores(y_pred_list_test, y_true_list_test, 2)
    y_true_test_true_only = np.array(y_true_test)[:, 0]

    test_qwk = (cohen_kappa_score(y_pred_test, y_true_test_true_only, weights='quadratic'))

    if plot:    
        plt.title(nn_name + " Val and Test QWK for Inference")
        plt.bar("val", val_qwk, label="Val QWK")
        plt.text(0, val_qwk / 2, val_qwk, ha='center')
        plt.bar("test", test_qwk, label="Testing QWK")
        plt.text(1, test_qwk / 2, test_qwk, ha='center')
        plt.ylabel('QWK')
        plt.xlabel('Epoch')
        plt.legend()

        plt.show()

    print(nn_name + " Val and Test QWK for Inference:")
    print("Val QWK: " + str(val_qwk))
    print("Test QWK: " + str(test_qwk))

def load_data_for_inference():
    try:
        data = pickle.load(open('NeuralNetworks/inference_data.pkl', 'rb'))
        print("found file")

        return data["val"], data["test"]

    except:
        _, _, X_val, Y_val, X_test, Y_test = load_everything_navid()

        val = {"X_val": X_val[200:700], "Y_val": Y_val[200:700]}
        test = {"X_test": X_test[400:900], "Y_test": Y_test[400:900]}


        data = {"val": val, "test": test}

        with open('NeuralNetworks/inference_data.pkl', 'wb') as f:
            pickle.dump(data, f)

        return val, test

def load_model_run_inference():
    print("device is " + str(torch.device))

    val, test = load_data_for_inference()

    random_number = np.random.randint(0, 200)

    val_loader = create_loader(val["X_val"][random_number:random_number+250], val["Y_val"][random_number:random_number+250])
    test_loader = create_loader(test["X_test"][random_number:random_number+250], test["Y_test"][random_number:random_number+250])

    mlp = torch.load("NeuralNetworks/MLP_model", weights_only=False)
    mlp.eval()

    y_pred_list_val, y_true_list_val = mlp.test_model(val_loader)
    y_pred_list_test, y_true_list_test = mlp.test_model(test_loader)

    plot = False

    inference_data_visualization(mlp.get_name(), y_pred_list_val, y_true_list_val, y_pred_list_test, y_true_list_test, plot)

    bnn = torch.load("NeuralNetworks/BNN_model", weights_only=False)
    bnn.eval()

    y_pred_list_val, y_true_list_val = bnn.test_model(val_loader)
    y_pred_list_test, y_true_list_test = bnn.test_model(test_loader)

    inference_data_visualization(bnn.get_name(), y_pred_list_val, y_true_list_val, y_pred_list_test, y_true_list_test, plot)


def main():
    set_up_train_test_plot()
    

if __name__=="__main__":
    main()