import preprocessing
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, cohen_kappa_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, uniform
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
import numpy as np
import os
import pickle
import sys
from tqdm import tqdm
import torch
import seaborn as sns
from sklearn.metrics import confusion_matrix

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU detected: {gpu_name}")
else:
    print("GPU not available, using CPU")

sentence_model = SentenceTransformer("all-mpnet-base-v2", device=device)

def get_document_embedding(essay):
    """
    Gets essay level embedding by encoding the entire essay as a single string using SentenceTransformer.
    """
    embedding = sentence_model.encode(essay)
    
    return embedding

def data_tokenize(asap1_train, asap1_val, asap2, batch_size=32):
    """
    Generates embeddings for all essays in asap1_train, asap1_val, and asap2 datasets using SentenceTransformer.
    Returns a dictionary containing embeddings and normalized scores for each dataset. Embeddings are generated in batches with a progress bar for efficiency.
    """
    all_texts = list(asap1_train['text']) + list(asap1_val['text']) + list(asap2['text'])
    
    all_embeddings = []
    for i in tqdm(range(0, len(all_texts), batch_size), desc="Generating embeddings", unit="batch"):
        batch_texts = all_texts[i:i + batch_size]
        batch_embeddings = sentence_model.encode(batch_texts, show_progress_bar=False, device=device)
        all_embeddings.extend(batch_embeddings)
    
    num_train = len(asap1_train)
    num_val = len(asap1_val)
    train_embeddings = all_embeddings[:num_train]
    val_embeddings = all_embeddings[num_train:num_train + num_val]
    asap2_embeddings = all_embeddings[num_train + num_val:]
    
    train_scores = list(asap1_train['norm_score'])
    val_scores = list(asap1_val['norm_score'])
    asap2_scores = list(asap2['norm_score'])
    
    return {
        'asap1_train': {'embeddings': train_embeddings, 'scores': train_scores},
        'asap1_val': {'embeddings': val_embeddings, 'scores': val_scores},
        'asap2': {'embeddings': asap2_embeddings, 'scores': asap2_scores}
    }

def embed_and_pickle():
    '''
    The embedding file used in here is the embedding file which was used to train the model in this.
    This file may be missing from the github.
    In order to get the missing file, perhaps look into the NeuralNetworks code; I used Alex's embeddings for consistency. 
    '''
    with open('processed_data/alex_embeddings_no_prompt_with_demographics.pkl', 'rb') as f:
        data = pickle.load(f)
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

def evaluate_model(y_true_norm, y_pred_norm, dataset_name, y_true_raw=None, y_pred_raw=None):
    '''
    Evaluates the model's performance by calculating MSE, MAE, Accuracy, and Quadratic Weighted Kappa.
    '''
    mse = mean_squared_error(y_true_norm, y_pred_norm)
    mae = mean_absolute_error(y_true_norm, y_pred_norm)
    y_true_int = np.rint(y_true_norm)
    y_pred_int = np.rint(y_pred_norm)
    
    if y_true_raw is not None and y_pred_raw is not None:
        y_true_kappa = np.rint(y_true_raw)
        y_pred_kappa = np.rint(y_pred_raw)
    else:
        y_true_kappa = y_true_int
        y_pred_kappa = y_pred_int
    acc = accuracy_score(y_true_kappa, y_pred_kappa)
    kappa = cohen_kappa_score(y_true_kappa, y_pred_kappa, weights='quadratic')

    print(f"{dataset_name}:")
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  Accuracy: {acc:.4f} (rounded)")
    print(f"  Quadratic Weighted Kappa: {kappa:.4f}")
    print()
    
    return {'MSE': mse, 'MAE': mae, 'Accuracy': acc, 'Kappa': kappa}

def ensure_cache_dir():
    if not os.path.exists('processed_data'):
        os.makedirs('processed_data')

def save_tokenized_data(tokenized, cache_path='processed_data/embeddings.pkl'):
    ensure_cache_dir()
    with open(cache_path, 'wb') as f:
        pickle.dump(tokenized, f)
    print(f" Cached embeddings to {cache_path}")

def load_tokenized_data(cache_path='processed_data/embeddings.pkl'):
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            tokenized = pickle.load(f)
        print(f" Loaded embeddings from cache ({cache_path})")
        return tokenized
    return None

def scale_array(arr, scalar):
    return (arr * scalar).astype(int)

def get_results():
    '''
    For usage in comparemodels.py
    Returns the results of the model evaluation in a dictionary format.
    '''
    
    print("Loading data from pickle...")
    X_train, Y_train, X_val, Y_val, X_test, Y_test = embed_and_pickle()
    
    X_train = np.asarray(X_train)
    y_train = np.asarray(Y_train[:, 0])
    
    X_val = np.asarray(X_val)
    y_val = np.asarray(Y_val[:, 0])
    
    X_asap2 = np.asarray(X_test)
    y_asap2 = np.asarray(Y_test[:, 0])
    
    train_prompt_ids = np.asarray(Y_train[:, 1]) 
    val_prompt_ids = np.asarray(Y_val[:, 1])
    asap2_prompt_ids = np.asarray(Y_test[:, 1])
    param_dist = {
        'alpha': loguniform(1e-3, 1e3),          
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
        'fit_intercept': [True, False],
        'tol': loguniform(1e-4, 1e-1),            
    }
    
    
    base_model = Ridge(random_state=42)
    
   
    print("\nTuning LinearRegression (Ridge) hyperparameters with RandomizedSearchCV...")
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=50,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    
    random_search.fit(X_train, y_train)
    
    
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    best_val_mse = mean_squared_error(y_val, best_model.predict(X_val))
    best_model_name = f"Ridge (optimized)"
    
    print("\n" + "=" * 60)
    print(f"BEST HYPERPARAMETERS:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"Best CV MSE: {-random_search.best_score_:.4f}")
    print("=" * 60)
    
    
    results = {}
    
    
    y_train_pred = best_model.predict(X_train)
    y_val_pred = best_model.predict(X_val)
    y_asap2_pred = best_model.predict(X_asap2)
    
    
    print(type(Y_train))
    print(Y_train)
    y_train_true_raw = np.array([preprocessing.denormalize_asap1(n, pid)
        for n, pid in zip(np.array(Y_train[:,0]), train_prompt_ids)])
    y_train_pred_raw = np.array([
        preprocessing.denormalize_asap1(n, pid)
        for n, pid in zip(y_train_pred, train_prompt_ids)
    ])
    y_val_true_raw = np.array([preprocessing.denormalize_asap1(n, pid)
        for n, pid in zip(np.array(Y_val[:,0]), val_prompt_ids)])
    y_val_pred_raw = np.array([
        preprocessing.denormalize_asap1(n, pid)
        for n, pid in zip(y_val_pred, val_prompt_ids)
    ])
    y_asap2_true_raw = np.array([preprocessing.denormalize_asap2(n)
        for n in np.array(Y_test[:,0])])
    y_asap2_pred_raw = np.array([
        int(preprocessing.denormalize_asap2(n)) for n in y_asap2_pred
    ])
    

    print(f"\nResults ({best_model_name}):")
    print("=" * 60)
    results['Training Set'] = evaluate_model(y_train, y_train_pred, "Training Set", y_train_true_raw, y_train_pred_raw)
    results['Validation Set'] = evaluate_model(y_val, y_val_pred, "Validation Set", y_val_true_raw, y_val_pred_raw)
    results['ASAP 2.0 Set'] = evaluate_model(y_asap2, y_asap2_pred, "ASAP 2.0 Set", y_asap2_true_raw, y_asap2_pred_raw)
    return results

if __name__ == '__main__':
    use_cache = '--cache' in sys.argv
    regenerate = '--regenerate' in sys.argv
    
    print("=" * 60)
    print("Essay Scoring Model")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Cache mode: use_cache={use_cache}, regenerate={regenerate}\n")
    
    print("Loading data from pickle...")
    X_train, Y_train, X_val, Y_val, X_test, Y_test = embed_and_pickle()
    
    print("Loading original data for raw scores...")
    data_array = preprocessing.load_all_data()
    
    X_train = np.asarray(X_train)
    y_train = np.asarray(Y_train[:, 0])  
    
    X_val = np.asarray(X_val)
    y_val = np.asarray(Y_val[:, 0])
    
    X_asap2 = np.asarray(X_test)
    y_asap2 = np.asarray(Y_test[:, 0])
    
    train_prompt_ids = np.asarray(Y_train[:, 1]) 
    val_prompt_ids = np.asarray(Y_val[:, 1])
    asap2_prompt_ids = np.asarray(Y_test[:, 1])  
    param_dist = {
        'alpha': loguniform(1e-3, 1e3),          
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
        'fit_intercept': [True, False],
        'tol': loguniform(1e-4, 1e-1),            
    }
    
   
    base_model = Ridge(random_state=42)
    
    print("\nTuning LinearRegression (Ridge) hyperparameters with RandomizedSearchCV...")
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=50,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    random_search.fit(X_train, y_train)
    
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    best_val_mse = mean_squared_error(y_val, best_model.predict(X_val))
    best_model_name = f"Ridge (optimized)"
    
    print("\n" + "=" * 60)
    print(f"BEST HYPERPARAMETERS:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"Best CV MSE: {-random_search.best_score_:.4f}")
    print("=" * 60)
    
    results = {}
    
    y_train_pred = best_model.predict(X_train)
    y_val_pred = best_model.predict(X_val)
    y_asap2_pred = best_model.predict(X_asap2)
    
    print(type(Y_train))
    print(Y_train)
    y_train_true_raw = np.array([preprocessing.denormalize_asap1(n, pid)
        for n, pid in zip(np.array(Y_train[:,0]), train_prompt_ids)])
    y_train_pred_raw = np.array([
        preprocessing.denormalize_asap1(n, pid)
        for n, pid in zip(y_train_pred, train_prompt_ids)
    ])
    y_val_true_raw = np.array([preprocessing.denormalize_asap1(n, pid)
        for n, pid in zip(np.array(Y_val[:,0]), val_prompt_ids)])
    y_val_pred_raw = np.array([
        preprocessing.denormalize_asap1(n, pid)
        for n, pid in zip(y_val_pred, val_prompt_ids)
    ])
    y_asap2_true_raw = np.array([preprocessing.denormalize_asap2(n)
        for n in np.array(Y_test[:,0])])
    y_asap2_pred_raw = np.array([
        int(preprocessing.denormalize_asap2(n)) for n in y_asap2_pred
    ])
    
    print(f"\nResults ({best_model_name}):")
    print("=" * 60)
    results['Training Set'] = evaluate_model(y_train, y_train_pred, "Training Set", y_train_true_raw, y_train_pred_raw)
    results['Validation Set'] = evaluate_model(y_val, y_val_pred, "Validation Set", y_val_true_raw, y_val_pred_raw)
    results['ASAP 2.0 Set'] = evaluate_model(y_asap2, y_asap2_pred, "ASAP 2.0 Set", y_asap2_true_raw, y_asap2_pred_raw)
    
    print("\nPer-Prompt Results on ASAP1 Validation Set:")
    print("=" * 60)
    # per prompt evaluation on ASAP 1.0 validation set
    unique_prompts = sorted(np.unique(val_prompt_ids))
    per_prompt_results = {}
    for pid in unique_prompts:
        mask = val_prompt_ids == pid
        y_true_norm = y_val[mask]
        y_pred_norm = y_val_pred[mask]
        y_true_raw = y_val_true_raw[mask]
        y_pred_raw = y_val_pred_raw[mask]
        per_prompt_results[f"Prompt {pid}"] = evaluate_model(y_true_norm, y_pred_norm, f"Prompt {pid}", y_true_raw, y_pred_raw)
    
    # demographic analysis on ASAP 2.0
    print("\nSubgroup Results on ASAP 2.0 Set:")
    print("=" * 60)
    
    value_maps = {
        'economically_disadvantaged': {1: "Economically disadvantaged", 0: "Not economically disadvantaged"},
        'student_disability_status': {1: "Identified as having disability", 0: "Not identified as having disability"},
        'ell_status': {1: "Yes", 0: "No"},
        'gender': {1: "M", 0: "F"},
        'race_ethnicity': {0: "American Indian/Alaskan Native", 1: "Asian/Pacific Islander", 2: "Black/African American", 3: "Hispanic/Latino", 4: "Two or more races/Other", 5: "White"}
    }
    
    subgroups = {
        'economically_disadvantaged': 2,
        'student_disability_status': 3,
        'ell_status': 4,
        'race_ethnicity': 5,
        'gender': 6
    }
    subgroup_results = {}
    for name, idx in subgroups.items():
        val_map = value_maps[name]
        unique_vals = np.unique(Y_test[:, idx])
        subgroup_results[name] = {}
        for val in unique_vals:
            if val not in val_map:
                continue
            label = val_map[val]
            mask = Y_test[:, idx] == val
            if mask.sum() == 0:
                continue
            y_true_norm = y_asap2[mask]
            y_pred_norm = y_asap2_pred[mask]
            y_true_raw = y_asap2_true_raw[mask]
            y_pred_raw = y_asap2_pred_raw[mask]
            subgroup_results[name][label] = evaluate_model(y_true_norm, y_pred_norm, f"{name}={label}", y_true_raw, y_pred_raw)
    
    print("\nGenerating per-prompt visualizations...")
    prompts = list(per_prompt_results.keys())
    metrics = ['MSE', 'MAE', 'Accuracy', 'Kappa']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Per-Prompt Model Performance on ASAP1 Validation Set\n{best_model_name}', fontsize=14, fontweight='bold')
    
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        x = np.arange(len(prompts))
        width = 0.6
        
        values = [per_prompt_results[prompt].get(metric, 0) for prompt in prompts]
        
        bars = ax.bar(x, values, width, color='steelblue', alpha=0.8)
        
 
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Prompt')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} Across Prompts')
        ax.set_xticks(x)
        ax.set_xticklabels(prompts, rotation=15, ha='right')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('processed_data/per_prompt_comparison_noprompt.png', dpi=300, bbox_inches='tight')
    print(" Saved per-prompt visualization to processed_data/per_prompt_comparison_noprompt.png")
    plt.show()
    
    print("\nGenerating visualizations...")
    datasets = ['Training Set', 'Validation Set', 'ASAP 2.0 Set']
    metrics = ['MSE', 'MAE', 'Accuracy', 'Kappa']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Best LinearRegression Model Performance\n{best_model_name}', fontsize=14, fontweight='bold')
    
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        x = np.arange(len(datasets))
        width = 0.6
        
        values = [results[dataset].get(metric, 0) for dataset in datasets]
        
        bars = ax.bar(x, values, width, color='steelblue', alpha=0.8)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Dataset')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} Across Datasets')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=15, ha='right')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('processed_data/model_comparison_noprompt.png', dpi=300, bbox_inches='tight')
    print(" Saved visualization to processed_data/model_comparison_noprompt.png")
    plt.show()
    
    print("\nGenerating subgroup visualizations...")
    for name, sub_results in subgroup_results.items():
        if not sub_results:
            continue
        subgroups_list = list(sub_results.keys())
        metrics = ['MSE', 'MAE', 'Accuracy', 'Kappa']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Model Performance by {name}\n{best_model_name}', fontsize=14, fontweight='bold')
        
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            x = np.arange(len(subgroups_list))
            width = 0.6
            
            values = [sub_results[sub].get(metric, 0) for sub in subgroups_list]
            
            bars = ax.bar(x, values, width, color='steelblue', alpha=0.8)
    
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}',
                       ha='center', va='bottom', fontsize=9)
            
            ax.set_xlabel(name)
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} by {name}')
            ax.set_xticks(x)
            ax.set_xticklabels(subgroups_list, rotation=15, ha='right')
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'processed_data/subgroup_{name}_comparison_noprompt.png', dpi=300, bbox_inches='tight')
        print(f" Saved {name} visualization to processed_data/subgroup_{name}_comparison_noprompt.png")
        plt.show()
    
    # generate confusion matrices for ASAP 2.0
    print("\nGenerating confusion matrix for ASAP 2.0...")
    cm = confusion_matrix(y_asap2_true_raw, y_asap2_pred_raw)

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm_norm,
        annot=cm,           
        fmt='d',
        cmap='Blues',
        vmin=0,
        vmax=1,
        ax=ax
    )

    ax.set_xlabel('Predicted Score')
    ax.set_ylabel('True Score')
    ax.set_title('Confusion Matrix - ASAP 2.0 Set')

    unique_scores = sorted(np.unique(np.concatenate([y_asap2_true_raw, y_asap2_pred_raw])))
    ax.set_xticklabels(unique_scores)
    ax.set_yticklabels(unique_scores)

    plt.tight_layout()
    plt.savefig('processed_data/confusion_matrix_noprompt.png', dpi=300, bbox_inches='tight')
    print(" Saved confusion matrix to processed_data/confusion_matrix_noprompt.png")
    plt.show()
    
    # generate confusion matrices for ASAP 1.0 (combining train and val sets)
    # the resulting matrix is quite large due to the different score ranges of essay sets
    # you should mainly look at scores from 0-6
    print("\nGenerating confusion matrix for ASAP 1.0...")
    combined_ASAP1_true = np.concatenate([y_train_true_raw, y_val_true_raw])
    combined_ASAP1_pred = np.concatenate([y_train_pred_raw, y_val_pred_raw])
    cm = confusion_matrix(combined_ASAP1_true, combined_ASAP1_pred)

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm_norm,
        annot=cm,           
        fmt='d',
        cmap='Blues',
        vmin=0,
        vmax=1,
        ax=ax
    )

    ax.set_xlabel('Predicted Score')
    ax.set_ylabel('True Score')
    ax.set_title('Confusion Matrix - ASAP 1.0 Set')

    unique_scores = sorted(np.unique(np.concatenate([combined_ASAP1_true, combined_ASAP1_pred])))
    ax.set_xticklabels(unique_scores)
    ax.set_yticklabels(unique_scores)

    plt.tight_layout()
    plt.savefig('processed_data/asap1_confusion_matrix_noprompt.png', dpi=300, bbox_inches='tight')
    print(" Saved confusion matrix to processed_data/asap1_confusion_matrix_noprompt.png")
    plt.show()
    
   