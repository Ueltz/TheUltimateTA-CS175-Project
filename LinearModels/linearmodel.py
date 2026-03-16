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

# Detect GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU detected: {gpu_name}")
else:
    print("GPU not available, using CPU")

sentence_model = SentenceTransformer("all-mpnet-base-v2", device=device)

def get_document_embedding(essay):
    """Encode entire document as a single embedding"""
    embedding = sentence_model.encode(essay)
    
    return embedding

def data_tokenize(asap1_train, asap1_val, asap2, batch_size=32):
    all_texts = list(asap1_train['text']) + list(asap1_val['text']) + list(asap2['text'])
    
    # Generate embeddings in batches with progress bar
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

def evaluate_model(y_true_norm, y_pred_norm, dataset_name, y_true_raw=None, y_pred_raw=None):
    """Compute metrics using normalized scores, except kappa uses raw values.

    Parameters:
    - y_true_norm, y_pred_norm: normalized [0,1] arrays for MSE/MAE/accuracy
    - y_true_raw, y_pred_raw: denormalized arrays (same length) for quadratic kappa
    """
    mse = mean_squared_error(y_true_norm, y_pred_norm)
    mae = mean_absolute_error(y_true_norm, y_pred_norm)
    # round normalized for accuracy
    y_true_int = np.rint(y_true_norm)
    y_pred_int = np.rint(y_pred_norm)
    

    # kappa source: raw if provided else rounded normalized
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
    """Create processed_data directory if it doesn't exist."""
    if not os.path.exists('processed_data'):
        os.makedirs('processed_data')

def save_tokenized_data(tokenized, cache_path='processed_data/embeddings.pkl'):
    """Save tokenized embeddings and scores to disk."""
    ensure_cache_dir()
    with open(cache_path, 'wb') as f:
        pickle.dump(tokenized, f)
    print(f"✓ Cached embeddings to {cache_path}")

def load_tokenized_data(cache_path='processed_data/embeddings.pkl'):
    """Load tokenized embeddings and scores from disk."""
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            tokenized = pickle.load(f)
        print(f"✓ Loaded embeddings from cache ({cache_path})")
        return tokenized
    return None

def scale_array(arr, scalar):
    """Scale a numpy array by a scalar value."""
    return (arr * scalar).astype(int)

def get_results():
    # always load data early (needed for prompt_ids and raw scores)
    print("Loading data...")
    data_array = preprocessing.load_all_data()

    # Try to load from cache first (if not regenerating)
    tokenized = load_tokenized_data()
    # Prepare training data
    X_train = np.array(tokenized['asap1_train']['embeddings'])
    y_train = np.array(tokenized['asap1_train']['scores'])
    # note: scores are already normalized to [0,1]
    
    # Prepare validation data
    X_val = np.array(tokenized['asap1_val']['embeddings'])
    y_val = np.array(tokenized['asap1_val']['scores'])
    # normalized scores kept as-is
    
    # Prepare ASAP2 data
    X_asap2 = np.array(tokenized['asap2']['embeddings'])
    y_asap2 = np.array(tokenized['asap2']['scores'])
    # normalized scores kept as-is
    
    # grab prompt ids from original data (for denormalization later)
    train_prompt_ids = np.array(data_array['asap1_train']['prompt_id'])
    val_prompt_ids = np.array(data_array['asap1_val']['prompt_id'])
    asap2_prompt_ids = np.array(data_array['asap2']['prompt_id'])
    
    # Define hyperparameter search space for Ridge regression
    # Using RandomizedSearchCV to tune all Ridge hyperparameters
    param_dist = {
        'alpha': loguniform(1e-3, 1e3),           # Regularization strength
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
        'fit_intercept': [True, False],
        'tol': loguniform(1e-4, 1e-1),            # Tolerance for convergence
    }
    
    # Initialize base Ridge model
    base_model = Ridge(random_state=42)
    
    # Create RandomizedSearchCV with 50 iterations
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
    
    # Fit RandomizedSearchCV on training data
    random_search.fit(X_train, y_train)
    
    # Get best model and parameters
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
    
    # Store results in dictionary format for consistency with visualization code
    results = {}
    
    # Evaluate best model on all datasets
    y_train_pred = best_model.predict(X_train)
    y_val_pred = best_model.predict(X_val)
    y_asap2_pred = best_model.predict(X_asap2)
    # Denormalize both predictions and true normalized values
    y_train_true_raw = np.array(data_array['asap1_train']['raw_score'], dtype=int)
    y_train_pred_raw = np.array([
        preprocessing.denormalize_asap1(n, pid)
        for n, pid in zip(y_train_pred, train_prompt_ids)
    ])
    y_val_true_raw = np.array(data_array['asap1_val']['raw_score'], dtype=int)
    y_val_pred_raw = np.array([
        preprocessing.denormalize_asap1(n, pid)
        for n, pid in zip(y_val_pred, val_prompt_ids)
    ])
    y_asap2_true_raw = np.array(data_array['asap2']['raw_score'], dtype=int)
    y_asap2_pred_raw = np.array([
        preprocessing.denormalize_asap2(n) for n in y_asap2_pred
    ])
    
    # Evaluate the best model on all datasets
    print(f"\nResults ({best_model_name}):")
    print("=" * 60)
    results['Training Set'] = evaluate_model(y_train, y_train_pred, "Training Set", y_train_true_raw, y_train_pred_raw)
    results['Validation Set'] = evaluate_model(y_val, y_val_pred, "Validation Set", y_val_true_raw, y_val_pred_raw)
    results['ASAP 2.0 Set'] = evaluate_model(y_asap2, y_asap2_pred, "ASAP 2.0 Set", y_asap2_true_raw, y_asap2_pred_raw)
    return results

if __name__ == '__main__':
    # Check for --cache flag
    use_cache = '--cache' in sys.argv
    regenerate = '--regenerate' in sys.argv
    
    print("=" * 60)
    print("Essay Scoring Model")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Cache mode: use_cache={use_cache}, regenerate={regenerate}\n")
    
    # always load data early (needed for prompt_ids and raw scores)
    print("Loading data...")
    data_array = preprocessing.load_all_data()

    # Try to load from cache first (if not regenerating)
    tokenized = None
    if use_cache and not regenerate:
        tokenized = load_tokenized_data()

    # If not loaded from cache, generate embeddings
    if tokenized is None:
        print("\nGenerating embeddings (this may take a few minutes)...")
        tokenized = data_tokenize(data_array['asap1_train'], data_array['asap1_val'], data_array['asap2'])
        # Save to cache if requested
        if use_cache:
            save_tokenized_data(tokenized)
    
    
    # Prepare training data
    X_train = np.array(tokenized['asap1_train']['embeddings'])
    y_train = np.array(tokenized['asap1_train']['scores'])
    # note: scores are already normalized to [0,1]
    
    # Prepare validation data
    X_val = np.array(tokenized['asap1_val']['embeddings'])
    y_val = np.array(tokenized['asap1_val']['scores'])
    # normalized scores kept as-is
    
    # Prepare ASAP2 data
    X_asap2 = np.array(tokenized['asap2']['embeddings'])
    y_asap2 = np.array(tokenized['asap2']['scores'])
    # normalized scores kept as-is
    
    # grab prompt ids from original data (for denormalization later)
    train_prompt_ids = np.array(data_array['asap1_train']['prompt_id'])
    val_prompt_ids = np.array(data_array['asap1_val']['prompt_id'])
    asap2_prompt_ids = np.array(data_array['asap2']['prompt_id'])
    
    # Define hyperparameter search space for Ridge regression
    # Using RandomizedSearchCV to tune all Ridge hyperparameters
    param_dist = {
        'alpha': loguniform(1e-3, 1e3),           # Regularization strength
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
        'fit_intercept': [True, False],
        'tol': loguniform(1e-4, 1e-1),            # Tolerance for convergence
    }
    
    # Initialize base Ridge model
    base_model = Ridge(random_state=42)
    
    # Create RandomizedSearchCV with 50 iterations
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
    
    # Fit RandomizedSearchCV on training data
    random_search.fit(X_train, y_train)
    
    # Get best model and parameters
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
    
    # Store results in dictionary format for consistency with visualization code
    results = {}
    
    # Evaluate best model on all datasets
    y_train_pred = best_model.predict(X_train)
    y_val_pred = best_model.predict(X_val)
    y_asap2_pred = best_model.predict(X_asap2)
    # Denormalize both predictions and true normalized values
    y_train_true_raw = np.array(data_array['asap1_train']['raw_score'], dtype=int)
    y_train_pred_raw = np.array([
        preprocessing.denormalize_asap1(n, pid)
        for n, pid in zip(y_train_pred, train_prompt_ids)
    ])
    y_val_true_raw = np.array(data_array['asap1_val']['raw_score'], dtype=int)
    y_val_pred_raw = np.array([
        preprocessing.denormalize_asap1(n, pid)
        for n, pid in zip(y_val_pred, val_prompt_ids)
    ])
    y_asap2_true_raw = np.array(data_array['asap2']['raw_score'], dtype=int)
    y_asap2_pred_raw = np.array([
        preprocessing.denormalize_asap2(n) for n in y_asap2_pred
    ])
    
    # Evaluate the best model on all datasets
    print(f"\nResults ({best_model_name}):")
    print("=" * 60)
    results['Training Set'] = evaluate_model(y_train, y_train_pred, "Training Set", y_train_true_raw, y_train_pred_raw)
    results['Validation Set'] = evaluate_model(y_val, y_val_pred, "Validation Set", y_val_true_raw, y_val_pred_raw)
    results['ASAP 2.0 Set'] = evaluate_model(y_asap2, y_asap2_pred, "ASAP 2.0 Set", y_asap2_true_raw, y_asap2_pred_raw)
    
    # Per-prompt evaluation on ASAP1 validation set
    print("\nPer-Prompt Results on ASAP1 Validation Set:")
    print("=" * 60)
    unique_prompts = sorted(data_array['asap1_val']['prompt_id'].unique())
    per_prompt_results = {}
    for pid in unique_prompts:
        mask = val_prompt_ids == pid
        y_true_norm = y_val[mask]
        y_pred_norm = y_val_pred[mask]
        y_true_raw = y_val_true_raw[mask]
        y_pred_raw = y_val_pred_raw[mask]
        per_prompt_results[f"Prompt {pid}"] = evaluate_model(y_true_norm, y_pred_norm, f"Prompt {pid}", y_true_raw, y_pred_raw)
    
    # Create per-prompt visualizations
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
        
        # Add value labels on bars
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
    plt.savefig('processed_data/per_prompt_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved per-prompt visualization to processed_data/per_prompt_comparison.png")
    plt.show()
    
    # Create visualizations
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
        
        # Add value labels on bars
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
    plt.savefig('processed_data/model_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved visualization to processed_data/model_comparison.png")
    plt.show()
    
    # Confusion matrix for ASAP 2.0
    print("\nGenerating confusion matrix for ASAP 2.0...")
    # Round predictions to nearest integer for confusion matrix
    y_asap2_pred_raw_rounded = np.round(y_asap2_pred_raw).astype(int)
    cm = confusion_matrix(y_asap2_true_raw, y_asap2_pred_raw_rounded)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted Score')
    ax.set_ylabel('True Score')
    ax.set_title('Confusion Matrix - ASAP 2.0 Set')
    # Set tick labels based on actual unique scores
    unique_scores = sorted(np.unique(np.concatenate([y_asap2_true_raw, y_asap2_pred_raw_rounded])))
    ax.set_xticklabels(unique_scores)
    ax.set_yticklabels(unique_scores)
    plt.tight_layout()
    plt.savefig('processed_data/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Saved confusion matrix to processed_data/confusion_matrix.png")
    plt.show()