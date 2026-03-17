"""
This file is responsible for accepting data from the neural networks and displaying them for analysis. 
It is capable of showing general statistics (loss, functions, accuracy, qwk) per epoch, plots qwk per essay per epoch, confusion matrices (per essay set and overall), 
and average scores categorized by race/ethnicity (per essay set and overall, only applicable to asap2). 

Written By: Alexander Lenz
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, mean_absolute_error, mean_squared_error, accuracy_score, cohen_kappa_score
from Data_processing import denormalize_all_scores

#separates the data by race/ethnicity
def separate_by_race_ethnicity(y_pred_list, y_true_list):
    y_pred_list_by_race_ethnicity = []
    y_true_list_by_race_ethnicity = []

    for i in range(6):
        y_pred_list_by_race_ethnicity.append([])
        y_true_list_by_race_ethnicity.append([])

    for i in range(len(y_pred_list)):
        race_ethnicity = int(y_true_list[i][5])
        
        y_pred_list_by_race_ethnicity[race_ethnicity].append(y_pred_list[i])
        y_true_list_by_race_ethnicity[race_ethnicity].append(y_true_list[i][0])


    return y_pred_list_by_race_ethnicity, y_true_list_by_race_ethnicity

#plots bargraphs of average score per race/ethnicity, predicted and true
def plot_by_race_ethnicity(nn_name, y_pred_list_test, y_true_list_test): #only accepts asap 2

    essay_sets_by_race_ethnicity_pred = []
    essay_sets_by_race_ethnicity_test = []

    races = ["American Indian/Alaskan Native", "Asian/Pacific Islander", "Black/African American", "Hispanic/Latino", "Two or more races/Other", "White"]

    test_totals = [0, 0, 0, 0, 0, 0]
    pred_totals = [0, 0, 0, 0, 0, 0]
    test_total_len = [0, 0, 0, 0, 0, 0]
    pred_total_len = [0, 0, 0, 0, 0, 0]

 
    for i in range(len(y_true_list_test)):

        fig = plt.gcf()
        fig.set_size_inches(15, 10)

        y_true_test_set = np.array(y_true_list_test[i][-1])
        y_pred_test_set = np.array(y_pred_list_test[i][-1])

        y_pred_list_by_race_ethnicity, y_true_list_by_race_ethnicity = separate_by_race_ethnicity(y_pred_test_set, y_true_test_set)

        essay_sets_by_race_ethnicity_pred.append(y_pred_list_by_race_ethnicity)
        essay_sets_by_race_ethnicity_test.append(y_true_list_by_race_ethnicity)


        test_averages = []
        pred_averages = []

        

        for j in range(len(essay_sets_by_race_ethnicity_test[i])):
            test_averages.append(sum(essay_sets_by_race_ethnicity_test[i][j])/len(essay_sets_by_race_ethnicity_test[i][j]))
            pred_averages.append(sum(essay_sets_by_race_ethnicity_pred[i][j])/len(essay_sets_by_race_ethnicity_pred[i][j]))
            test_totals[j] += sum(essay_sets_by_race_ethnicity_test[i][j])
            pred_totals[j] += sum(essay_sets_by_race_ethnicity_pred[i][j])
            test_total_len[j] += len(essay_sets_by_race_ethnicity_test[i][j])
            pred_total_len[j] += len(essay_sets_by_race_ethnicity_pred[i][j])

        for j in range(len(test_averages)):
            plt.bar(j, test_averages[j], label="predicted" if j == 0 else None, color='blue', alpha=0.5)
            plt.bar(j, pred_averages[j], label="true" if j == 0 else None, color='red', alpha=0.5)
            print("Set: " + str(i + 1) + " " + races[j] + " average predicted score: " + str(pred_averages[j]) + " average true score: " + str(test_averages[j]) + " difference: " + str(pred_averages[j] - test_averages[j]))

        print()
        plt.xticks([0, 1, 2, 3, 4, 5], races, rotation=25)
        plt.title(nn_name + " Scores for Race/Ethnicity set " + str(i + 1))
        plt.ylabel('Race/Ethnicity')
        plt.xlabel('Score')
        plt.legend()

        #plt.show()
        plt.savefig("NeuralNetworks/Figures/" + nn_name + "_plot_by_race_ethnicity_set" + str(i + 1) + ".png")
        plt.close(fig)

    fig = plt.gcf()
    fig.set_size_inches(15, 10)

    for i in range(len(test_averages)):
        test_race_average = test_totals[i]/test_total_len[i]
        predrace_average = pred_totals[i]/pred_total_len[i]
        plt.bar(i, test_race_average, label="predicted" if i == 0 else None, color='blue', alpha=0.5)
        plt.bar(i, predrace_average, label="true" if i == 0 else None, color='red', alpha=0.5)
        print(races[i] + " total average predicted score: " + str(predrace_average) + " total average true score: " + str(test_race_average) + " difference: " + str(predrace_average - test_race_average))

    print()
    plt.xticks([0, 1, 2, 3, 4, 5], races, rotation=25)
    plt.title(nn_name + " Scores for Race/Ethnicity set total")
    plt.ylabel('Race/Ethnicity')
    plt.xlabel('Score')
    plt.legend()

    #plt.show()
    plt.savefig("NeuralNetworks/Figures/" + nn_name + "_plot_by_race_ethnicity_total.png")
    plt.close(fig)
    
    pred_average = sum(pred_totals)/sum(pred_total_len)
    test_average = sum(test_totals)/sum(test_total_len)
    print("Asap2 total average predicted score: " + str(pred_average) + " total average true score: " + str(test_average) + " difference: " + str(pred_average - test_average))

    
#separates prompts by set
def separate_by_prompt_id(y_pred_list, y_true_list, num_of_prompts):
    y_pred_list_by_prompt_id = []
    y_true_list_by_prompt_id = []

    for i in range(num_of_prompts):
        y_pred_list_by_prompt_id.append([])
        y_true_list_by_prompt_id.append([])

    for i in range(len(y_pred_list)):
        for j in range(num_of_prompts):
            y_pred_list_by_prompt_id[j].append([])
            y_true_list_by_prompt_id[j].append([])

        for j in range(len(y_true_list[i])):
            prompt_id = (int)(y_true_list[i][j][1] - 1)
            
            if prompt_id < 8:
                y_pred_list_by_prompt_id[prompt_id][i].append((y_pred_list[i][j]))
                y_true_list_by_prompt_id[prompt_id][i].append((y_true_list[i][j]))
            else:
                prompt_id -= 100
                
                y_pred_list_by_prompt_id[prompt_id][i].append((y_pred_list[i][j]))
                y_true_list_by_prompt_id[prompt_id][i].append((y_true_list[i][j])) 

    return y_pred_list_by_prompt_id, y_true_list_by_prompt_id

#plots the qwk per essay set per epoch as well as some accuracy stats
def plot_per_essay(nn_name, asap1_pred_separate, asap1_true_separate, asap1_pred_separate_val, asap1_true_separate_val, asap2_pred_separate_test, asap2_true_separate_test):
    
    training_essays_qwk = []
    val_essays_qwk = []
    testing_essays_qwk = []

    for i in range(8):
        training_essays_qwk.append([])
        val_essays_qwk.append([])
        if i < 7:
            testing_essays_qwk.append([])

    fig, ax = plt.subplots(1, 3, figsize=(15, 12))

    for i in range(len(asap1_pred_separate)):
        for j in range(len(asap1_pred_separate[i])):
            
            training_essays_qwk[i].append(cohen_kappa_score(np.array(asap1_true_separate[i][j])[:, 0], asap1_pred_separate[i][j], weights='quadratic'))
            val_essays_qwk[i].append(cohen_kappa_score(np.array(asap1_true_separate_val[i][j])[:, 0], asap1_pred_separate_val[i][j], weights='quadratic'))
            if i < 7:
                testing_essays_qwk[i].append(cohen_kappa_score(np.array(asap2_true_separate_test[i][j])[:, 0], asap2_pred_separate_test[i][j], weights='quadratic'))


    ax[0].set_title(nn_name + " Training QWK Per Essay")
    for i in range(8):
        ax[0].plot(training_essays_qwk[i], label=f"Prompt " + str(i + 1), color=plt.cm.tab10(i))
    ax[0].set_ylabel('QWK')
    ax[0].set_xlabel('Epoch')
    ax[0].legend()

    ax[1].set_title(nn_name + " Validation QWK Per Essay")
    for i in range(8):
        ax[1].plot(val_essays_qwk[i], label=f"Prompt " + str(i + 1), color=plt.cm.tab10(i))
    ax[1].set_ylabel('QWK')
    ax[1].set_xlabel('Epoch')
    ax[1].legend()

    ax[2].set_title(nn_name + " Testing QWK Per Essay")
    for i in range(7):
        ax[2].plot(testing_essays_qwk[i], label=f"Prompt " + str(i + 1), color=plt.cm.tab10(i))
    ax[2].set_ylabel('QWK')
    ax[2].set_xlabel('Epoch')
    ax[2].legend()

    qwk_train_last = []
    qwk_val_last = []
    qwk_test_last = []

    for i in range(8):
        qwk_train_last.append(training_essays_qwk[i][-1])
        qwk_val_last.append(val_essays_qwk[i][-1])
        if i < 7:
            qwk_test_last.append(testing_essays_qwk[i][-1])


    qwk_train_sort = (np.argsort(qwk_train_last) + 1)[::-1]
    qwk_val_sort = (np.argsort(qwk_val_last) + 1)[::-1]
    qwk_test_sort = (np.argsort(qwk_test_last) + 1)[::-1]

    print(nn_name + " training qwk sorted: " + str(qwk_train_sort) + "\n")
    print(nn_name + " validation qwk sorted: " + str(qwk_val_sort) + "\n")
    print(nn_name + " testing qwk sorted: " + str(qwk_test_sort) + "\n")


    #plt.show()
    fig.savefig("NeuralNetworks/Figures/" + nn_name + "_per_essay_qwk.png")
    plt.close(fig)

#plots the confusion matrix for the last epoch
def plot_confusion_matrix(nn_name, y_pred_list_train, y_true_list_train, y_pred_list_val, y_true_list_val, y_pred_list_test, y_true_list_test):

    y_pred_train_rounded = list(map(round, np.array(y_pred_list_train[-1])))
    y_true_train_rounded = list(map(round, np.array(y_true_list_train[-1])[:, 0]))

    y_pred_val_rounded = list(map(round, np.array(y_pred_list_val[-1])))
    y_true_val_rounded = list(map(round, np.array(y_true_list_val[-1])[:, 0]))

    y_pred_test_rounded = list(map(round, np.array(y_pred_list_test[-1])))
    y_true_test_rounded = list(map(round, (np.array(y_true_list_test[-1]))[:, 0]))

    
    
    ConfusionMatrixDisplay.from_predictions(y_true_train_rounded, y_pred_train_rounded)
    fig = plt.gcf()
    fig.set_size_inches(15, 10)
    fig.tight_layout(pad=2.7)
    plt.title(nn_name + " Training Matrix All Sets")
    print("accuracy for training set: " + str(accuracy_score(y_true_train_rounded, y_pred_train_rounded)))
    
    #plt.show()
    fig.savefig("NeuralNetworks/Figures/" + nn_name + "_training_matrix_all_sets.png")
    plt.close(fig)

   

    
    ConfusionMatrixDisplay.from_predictions(y_true_val_rounded, y_pred_val_rounded)
    fig = plt.gcf()
    fig.set_size_inches(15, 10)
    fig.tight_layout(pad=2.7)
    plt.title(nn_name + " Validation Matrix All Sets")
    print("accuracy for validation set: " + str(accuracy_score(y_true_val_rounded, y_pred_val_rounded)))

    #plt.show()
    fig.savefig("NeuralNetworks/Figures/" + nn_name + "_validation_matrix_all_sets.png")
    plt.close(fig)
    
    

    ConfusionMatrixDisplay.from_predictions(y_true_test_rounded, y_pred_test_rounded)
    fig = plt.gcf()
    fig.set_size_inches(15, 10)
    fig.tight_layout(pad=2.7)
    plt.title(nn_name + " Testing Matrix All Sets")
    print("accuracy for testing set: " + str(accuracy_score(y_true_test_rounded, y_pred_test_rounded)))

    #plt.show()
    fig.savefig("NeuralNetworks/Figures/" + nn_name + "_testing_matrix_all_sets.png")
    plt.close(fig)

    
def plot_confusion_matrix_per_essay(nn_name, asap1_pred_separate, asap1_true_separate, asap1_pred_separate_val, asap1_true_separate_val, asap2_pred_separate_test, asap2_true_separate_test):

    for i in range(len(asap1_pred_separate)):
        ConfusionMatrixDisplay.from_predictions(np.array(asap1_true_separate[i][-1])[:, 0], asap1_pred_separate[i][-1])
        fig = plt.gcf()
        fig.set_size_inches(15, 10)
        fig.tight_layout(pad=2.7)
        plt.title(nn_name + " Training Matrix Set " + str(i + 1))
        print("accuracy for training set " + str(i + 1) + ": " + str(accuracy_score(np.array(asap1_true_separate[i][-1])[:, 0], asap1_pred_separate[i][-1])))
        
        #plt.show()
        fig.savefig("NeuralNetworks/Figures/" + nn_name + "_training_matrix_set_" + str(i + 1) + ".png")
        plt.close(fig)

   

    for i in range(len(asap1_pred_separate_val)):
        ConfusionMatrixDisplay.from_predictions(np.array(asap1_true_separate_val[i][-1])[:, 0], asap1_pred_separate_val[i][-1])
        fig = plt.gcf()
        fig.set_size_inches(15, 10)
        fig.tight_layout(pad=2.7)
        plt.title(nn_name + " Validation Matrix Set " + str(i + 1))
        print("accuracy for validation set " + str(i + 1) + ": " + str(accuracy_score(np.array(asap1_true_separate_val[i][-1])[:, 0], asap1_pred_separate_val[i][-1])))
        #plt.show()
        fig.savefig("NeuralNetworks/Figures/" + nn_name + "_validation_matrix_set_" + str(i + 1) + ".png")
        plt.close(fig)

    
    for i in range(len(asap2_pred_separate_test)):
        ConfusionMatrixDisplay.from_predictions(np.array(asap2_true_separate_test[i][-1])[:, 0], asap2_pred_separate_test[i][-1])
        fig = plt.gcf()
        fig.set_size_inches(15, 10)
        fig.tight_layout(pad=2.7)
        plt.title(nn_name + " Testing Matrix Set " + str(i + 1))
        print("accuracy for testing set " + str(i + 1) + ": " + str(accuracy_score(np.array(asap2_true_separate_test[i][-1])[:, 0], asap2_pred_separate_test[i][-1])))
        #plt.show()
        fig.savefig("NeuralNetworks/Figures/" + nn_name + "_testing_matrix_set_" + str(i + 1) + ".png")
        plt.close(fig)

#plots a slew of stats, qwk, accuracy, mse, mae, per epoch
def plot_general_stats(nn_name, y_pred_list_train_both, y_true_list_train_both, y_pred_list_val_both, y_true_list_val_both, y_pred_list_test_both, y_true_list_test_both):
    fig, ax = plt.subplots(2, 4, figsize=(15, 12))

    y_pred_list_train_denorm = y_pred_list_train_both[1]
    y_true_list_train_denorm = y_true_list_train_both[1]
    y_pred_list_val_denorm = y_pred_list_val_both[1]
    y_true_list_val_denorm = y_true_list_val_both[1]
    y_pred_list_test_denorm = y_pred_list_test_both[1]
    y_true_list_test_denorm = y_true_list_test_both[1]

    y_pred_list_train = y_pred_list_train_both[0]
    y_true_list_train = y_true_list_train_both[0]
    y_pred_list_val = y_pred_list_val_both[0]
    y_true_list_val = y_true_list_val_both[0]
    y_pred_list_test = y_pred_list_test_both[0]
    y_true_list_test = y_true_list_test_both[0]


    train_loss_logger = []
    val_loss_logger = []
    test_loss_logger = []

    val_accuracy_logger = []
    val_mae_logger = []
    val_mse_logger = []
    val_qwk_logger = []

    test_accuracy_logger = []
    test_mae_logger = []
    test_mse_logger = []
    test_qwk_logger = []



    for i in range(len(y_pred_list_val)):
        y_pred_train = y_pred_list_train[i]
        y_true_train = y_true_list_train[i]
        y_true_train_true_only = np.array(y_true_train)[:, 0]

        train_loss_logger.append(mean_squared_error(y_true_train_true_only, y_pred_train))

        y_pred_val = y_pred_list_val[i]
        y_true_val = y_true_list_val[i]
        y_true_val_true_only = np.array(y_true_val)[:, 0]

        val_loss_logger.append(mean_squared_error(y_true_val_true_only, y_pred_val))
        

        val_mse_logger.append(mean_squared_error(y_true_val_true_only, y_pred_val))
        val_mae_logger.append(mean_absolute_error(y_true_val_true_only, y_pred_val))


        y_pred_val_denorm = y_pred_list_val_denorm[i]
        y_true_val_denorm = y_true_list_val_denorm[i]
        y_true_val_true_only_denorm = np.array(y_true_val_denorm)[:, 0]

        y_pred_val_rounded = list(map(round, (np.array(y_pred_val_denorm))))
        y_true_val_rounded = list(map(round, (np.array(y_true_val_true_only_denorm))))

        val_accuracy_logger.append(accuracy_score(y_true_val_rounded, y_pred_val_rounded))
        val_qwk_logger.append(cohen_kappa_score(y_true_val_rounded, y_pred_val_rounded, weights='quadratic'))


        y_pred_test = y_pred_list_test[i]
        y_true_test = y_true_list_test[i]
        y_true_test_true_only = np.array(y_true_test)[:, 0]


        y_pred_test_denorm = y_pred_list_test_denorm[i]
        y_true_test_denorm = y_true_list_test_denorm[i]
        y_true_test_true_only_denorm = np.array(y_true_test_denorm)[:, 0]

        y_pred_test_rounded = list(map(round, (np.array(y_pred_test_denorm))))
        y_true_test_rounded = list(map(round, (np.array(y_true_test_true_only_denorm))))

        test_loss_logger.append(mean_squared_error(y_true_test_true_only, y_pred_test))

        test_accuracy_logger.append(accuracy_score(y_true_test_rounded, y_pred_test_rounded))
        
        test_mse_logger.append(mean_squared_error(y_true_test_true_only, y_pred_test))
        test_mae_logger.append(mean_absolute_error(y_true_test_true_only, y_pred_test))

        test_qwk_logger.append(cohen_kappa_score(y_true_test_rounded, y_pred_test_rounded, weights='quadratic'))


    print(nn_name + " train_loss_logger_list:\n" + str(train_loss_logger) + "\n")
    print(nn_name + " val_loss_logger_list:\n" + str(val_loss_logger) + "\n")
    print(nn_name + " val_accuracy_logger_list:\n" + str(val_accuracy_logger) + "\n")
    print(nn_name + " val_mae_logger_list:\n" + str(val_mae_logger) + "\n")
    print(nn_name + " val_mse_logger_list:\n" + str(val_mse_logger) + "\n")
    print(nn_name + " val_qwk_logger_list:\n" + str(val_qwk_logger) + "\n")

    print(nn_name + " test test_loss_logger_list:\n" + str(test_loss_logger) + "\n")
    print(nn_name + " test accuracy_logger_list:\n" + str(test_accuracy_logger) + "\n")
    print(nn_name + " test mae_logger_list:\n" + str(test_mae_logger) + "\n")
    print(nn_name + " test mse_logger_list:\n" + str(test_mse_logger) + "\n")
    print(nn_name + " test qwk_logger_list:\n" + str(test_qwk_logger) + "\n")


    print(nn_name + " min train loss:\n" + str(min(train_loss_logger)) + "\n")
    print(nn_name + " min val mae:\n" + str(min(val_mae_logger)) + "\n")
    print(nn_name + " min val mse:\n" + str(min(val_mse_logger)) + "\n")
    print(nn_name + " max val qwk:\n" + str(max(val_qwk_logger)) + "\n")
    print(nn_name + " max val accuracy:\n" + str(max(val_accuracy_logger)) + "\n")

    print(nn_name + " test min test loss:\n" + str(min(test_loss_logger)) + "\n")
    print(nn_name + " test min mae:\n" + str(min(test_mae_logger)) + "\n")
    print(nn_name + " test min mse:\n" + str(min(test_mse_logger)) + "\n")
    print(nn_name + " test max qwk:\n" + str(max(test_qwk_logger)) + "\n")
    print(nn_name + " test max accuracy:\n" + str(max(test_accuracy_logger)) + "\n")


    ax[0 * 2, 0].set_title(nn_name + " Training Loss vs Validation Loss")
    ax[0 * 2, 0].plot(train_loss_logger, label="Training Loss", color = 'blue')
    ax[0 * 2, 0].plot(val_loss_logger, label="Validation Loss", color='orange')
    ax[0 * 2, 0].set_ylabel('Loss')
    ax[0 * 2, 0].set_xlabel('Epoch')
    ax[0 * 2, 0].legend()

    
    ax[0 * 2, 1].set_title(nn_name + " Val Accuracy")
    ax[0 * 2, 1].plot(val_accuracy_logger, label="Testing Accuracy")
    ax[0 * 2, 1].set_ylabel('Accuracy')
    ax[0 * 2, 1].set_xlabel('Epoch')
    ax[0 * 2, 1].legend()

    ax[0 * 2, 2].set_title(nn_name + " Val MAE vs MSE")
    ax[0 * 2, 2].plot(val_mae_logger, label="Testing MAE", color='blue')
    ax[0 * 2, 2].plot(val_mse_logger, label="Testing MSE", color='orange')
    ax[0 * 2, 2].set_ylabel('Error')
    ax[0 * 2, 2].set_xlabel('Epoch')
    ax[0 * 2, 2].legend()

    ax[0 * 2, 3].set_title(nn_name + " Val QWK")
    ax[0 * 2, 3].plot(val_qwk_logger, label="Testing QWK")
    ax[0 * 2, 3].set_ylabel('QWK')
    ax[0 * 2, 3].set_xlabel('Epoch')
    ax[0 * 2, 3].legend()


    ax[0 * 2 + 1, 0].set_title(nn_name + " Testing Loss")
    ax[0 * 2 + 1, 0].plot(test_loss_logger, label="Asap2 Testing Loss", color='orange')
    ax[0 * 2 + 1, 0].set_ylabel('Loss')
    ax[0 * 2 + 1, 0].set_xlabel('Epoch')
    ax[0 * 2 + 1, 0].legend()

    
    ax[0 * 2 + 1, 1].set_title(nn_name + " Test Accuracy")
    ax[0 * 2 + 1, 1].plot(test_accuracy_logger, label="Asap2 Testing Accuracy")
    ax[0 * 2 + 1, 1].set_ylabel('Accuracy')
    ax[0 * 2 + 1, 1].set_xlabel('Epoch')
    ax[0 * 2 + 1, 1].legend()

    ax[0 * 2 + 1, 2].set_title(nn_name + " Test MAE vs MSE")
    ax[0 * 2 + 1, 2].plot(test_mae_logger, label="Asap2 Testing MAE", color='blue')
    ax[0 * 2 + 1, 2].plot(test_mse_logger, label="Asap2 Testing MSE", color='orange')
    ax[0 * 2 + 1, 2].set_ylabel('Error')
    ax[0 * 2 + 1, 2].set_xlabel('Epoch')
    ax[0 * 2 + 1, 2].legend()

    ax[0 * 2 + 1, 3].set_title(nn_name + " Test QWK")
    ax[0 * 2 + 1, 3].plot(test_qwk_logger, label="Asap2 Testing QWK")
    ax[0 * 2 + 1, 3].set_ylabel('QWK')
    ax[0 * 2 + 1, 3].set_xlabel('Epoch')
    ax[0 * 2 + 1, 3].legend()

    plt.subplots_adjust(wspace=0.5, hspace=0.5) 
    #plt.show()
    fig.savefig("NeuralNetworks/Figures/" + nn_name + "_general_stats.png")
    plt.close(fig)


    print("finished")

#calls all of the plotting functions
def plot_result_for_nn(nn_name, y_pred_list_train, y_true_list_train, y_pred_list_val, y_true_list_val, y_pred_list_test, y_true_list_test):
    y_pred_list_train_denorm, y_true_list_train_denorm, y_pred_list_val_denorm, y_true_list_val_denorm, y_pred_list_test_denorm, y_true_list_test_denorm = denormalize_all_scores(y_pred_list_train, y_true_list_train, y_pred_list_val, y_true_list_val, y_pred_list_test, y_true_list_test)

    asap1_pred_separate, asap1_true_separate = separate_by_prompt_id(y_pred_list_train_denorm, y_true_list_train_denorm, 8)
    asap1_pred_separate_val, asap1_true_separate_val = separate_by_prompt_id(y_pred_list_val_denorm, y_true_list_val_denorm, 8)
    asap2_pred_separate_test, asap2_true_separate_test = separate_by_prompt_id(y_pred_list_test_denorm, y_true_list_test_denorm, 7)

    plot_by_race_ethnicity(nn_name,asap2_pred_separate_test, asap2_true_separate_test)
    plot_confusion_matrix_per_essay(nn_name, asap1_pred_separate, asap1_true_separate, asap1_pred_separate_val, asap1_true_separate_val, asap2_pred_separate_test, asap2_true_separate_test)
    plot_confusion_matrix(nn_name, y_pred_list_train_denorm, y_true_list_train_denorm, y_pred_list_val_denorm, y_true_list_val_denorm, y_pred_list_test_denorm, y_true_list_test_denorm)
    plot_per_essay(nn_name, asap1_pred_separate, asap1_true_separate, asap1_pred_separate_val, asap1_true_separate_val, asap2_pred_separate_test, asap2_true_separate_test)
    plot_general_stats(nn_name, [y_pred_list_train, y_pred_list_train_denorm], [y_true_list_train, y_true_list_train_denorm], [y_pred_list_val, y_pred_list_val_denorm], [y_true_list_val, y_true_list_val_denorm], [y_pred_list_test, y_pred_list_test_denorm], [y_true_list_test, y_true_list_test_denorm])
    

def run_and_plot_nns(nns, train_loader, val_loader, test_loader):

    epoch = 60
    epochs = [epoch, epoch]

    for i in range(len(nns)):
        y_pred_list_train, y_true_list_train, y_pred_list_val, y_true_list_val, y_pred_list_test, y_true_list_test = nns[i].run_NN(train_loader, val_loader, test_loader, epochs[i])
        nn_name = nns[i].get_name()
        plot_result_for_nn(nn_name, y_pred_list_train, y_true_list_train, y_pred_list_val, y_true_list_val, y_pred_list_test, y_true_list_test)
        