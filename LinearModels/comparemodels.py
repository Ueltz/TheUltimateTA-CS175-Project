import numpy as np
import matplotlib.pyplot as plt
import linearmodel, linearmodelnoprompt

def plot_model_comparison(results_linear, results_no_prompt):
    datasets = list(results_linear.keys())
    metrics = ['MSE', 'MAE', 'Accuracy', 'Kappa']

    x = np.arange(len(metrics)) * 1.6
    width = 0.6

    fig, axes = plt.subplots(1, len(datasets), figsize=(18, 5), sharey=False)
    fig.suptitle("Model Performance Comparison", fontsize=14, fontweight='bold')

    for i, dataset in enumerate(datasets):
        ax = axes[i]

        vals_linear = [results_linear[dataset][m] for m in metrics]
        vals_noprompt = [results_no_prompt[dataset][m] for m in metrics]

        iP = ax.bar(x - width/2, vals_linear, width, label='Including Prompt')
        eP = ax.bar(x + width/2, vals_noprompt, width, label='Excluding Prompt')
        
        
        for bar in iP:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=7)
            
        for bar in eP:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_title(dataset)
        ax.set_ylabel("Score")
        ax.legend()

    plt.tight_layout()
    plt.savefig('processed_data/performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
if __name__ == "__main__":
    plot_model_comparison(linearmodel.get_results(), linearmodelnoprompt.get_results())