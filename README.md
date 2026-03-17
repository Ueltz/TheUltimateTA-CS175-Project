# The Ultimate TA Repository

External Libraries
Pytorch
TorchBnn
Pickle
Scikit-learn
Sentence_Transformers
Numpy
Matplotlib
Tqdm
Seaborn
nltk (in old versions)


Publicly Available Code
Alex:
https://www.youtube.com/watch?v=tJ3-KYMMOOs - (https://github.com/LukeDitria/pytorch_tutorials/blob/main/section03_pytorch_mlp/solutions/Pytorch1_MLP_Function_Approximation_Solution.ipynb)
I used this in conjugation with pytorch documentation to learn how the pytorch train and test loop works. The code has been heavily modified (pretty much every line) to suit my purposes. Remnants of this code can be found in NN.py, Mlp.py, and BNN.py.


Code Written Entirely By Your Team
Alex:
NN_main.py - This file pulls from all the other files to train, test, and visualize the results of the neural networks. It also contains code to run inferences which was created for the notebook. (140 lines)

Data_processing.py - This file is used in conjunction with the shared data prep file. This file extends shared data prep's capabilities to function on lists. It also loads the data for usage with the neural networks with the aid of shared data prep. It extracts features (removing prompts while doing so),
and remapping everything to integers to be used in tensors. Finally it embeds the essays and stores this data so it isn't reprocessed every time. (174 lines)

Analysis.py - This file is responsible for accepting data from the neural networks and displaying them for analysis. It is capable of showing general statistics (loss, functions, accuracy, qwk) per epoch, plots qwk per essay per epoch, confusion matrices (per essay set and overall), and average scores categorized by race/ethnicity (per essay set and overall, only applicable to asap2). (477 lines)

NN.py - This file contains the parent of MLP and BNN, NN. NN contains the train, and test loop, as well as logic for partially creating the neural networks. (116 lines)

MLP.py - This file contains the child of NN, MLP. MLP specifies what the NN is exactly by creating the layers, specifying the loss function, and setting the optimizer. (35 lines)

BNN.py - This file contains the child of NN, BNN. BNN specifies what the NN is exactly by creating the layers, specifying the loss function, and setting the optimizer. (37 lines)

Navid:
data.py - Rater averaging, normalization, quantile binning, text preprocessing, and loading datasets ASAP1 and ASAP2 (401 lines)

model.py - Longformer backbone with regression and ordinal head for zero-shot scoring (174 lines)

model_da.py - Takes a zero-shot model with a gradient reversal layer and domain discriminator for DANN adaptation. (234 lines)

losses.py - MSE, pairwise ranking, CORN ordinal, and soft QWK loss functions (247 lines)

train_supervised.py - Two-stage lora supervised training on ASAP1 with metric collection (505 lines)

train_adaptation.py - DANN + deep coral + uncertainty aware self training with dropout pseudo labeling (619 lines)

calibration.py - QWK threshold optimization and temp scaling for post hoc score calibration (96 lines)

evaluate.py - QWK computing, per prompt breakdowns, confusion matrix, significance testing, and error analysis. (178 lines)

run_pipeline.py - Main pipeline file for zeroshot, trains on ASAP1 and evaluates on ASAP2. (327 lines)

run_adaptation.py - Main file running domain adaptation. Loads checkpoint, runs DANN-Coral-UST, then evaluates on ASAP2 (352 lines)

setup_data.py - Verification for data being loaded correctly, and simple scripts to show rater stats and ranges (133 lines)

Ryan:
scores.py - Displays the distribution of scores for ASAP 2.0 for debugging purposes. (20 lines)

linearmodel.py - Performs training and evaluation of a linear model trained on the ASAP 1.0 dataset with the prompt attached. Contains functions which perform processing of the data after preprocessing.py to get essay-level embeddings. Additionally generates multiple graphs for analysis over prompts for ASAP 1.0, and a confusion matrix for ASAP 2.0. (453 lines)

linearmodelnoprompt.py - Performs training and evaluation of a linear model trained on the ASAP 1.0 dataset without the prompt attached. Uses Alex’s Data_processing.py (as in, loads embeddings generated from) to get the required embeddings / data for this model. Additionally generates multiple graphs for analysis over prompts for ASAP 1.0, and analysis over demographics for ASAP 2.0. Also generates confusion matrices for both ASAP 1.0 and ASAP 2.0. Note: a lot of the code in this file is extremely similar to linearmodel.py’s code – could have saved some lines by moving the functions into a separate file. (560 lines)

comparemodels.py - Compares the models from linearmodel.py and linearmodelnoprompt.py and generates a bar graph with side-by-side evaluation metrics over ASAP 1.0 and ASAP 2.0. (49 lines)

baseline.py - File used for testing which text representation would be best for the logistic regression model early on in the project. This is where the data in the presentation came from. Unfortunately, this file was not used in the final project. (102 lines)
