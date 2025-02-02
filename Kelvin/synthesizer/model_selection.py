import torch
from evaluation import compute_statistical_similarity, compute_downstream_performance
from opacus import PrivacyEngine

def evaluate_model(generator, data_loader, epsilon=1.0, privacy_engine=None):
    """
    Evaluate the generator model using utility metrics with DP noise.
    """
    # Get the data from the DataLoader's TensorDataset
    real_data = data_loader.dataset.tensors[0]  # Accessing the first tensor (features) from the TensorDataset
    synthetic_data = generator.generate(len(real_data))  # Assuming generator has a `generate` method

    # Convert synthetic data to NumPy if it's a tensor
    if isinstance(synthetic_data, torch.Tensor):
        synthetic_data = synthetic_data.detach().cpu().numpy()

    if isinstance(real_data, torch.Tensor):
        real_data = real_data.detach().cpu().numpy()

    # Ensure shape consistency
    if real_data.shape != synthetic_data.shape:
        raise ValueError(f"Shape mismatch: real_data ({real_data.shape}) vs synthetic_data ({synthetic_data.shape})")

    # Add DP noise during evaluation (if privacy_engine is provided)
    if privacy_engine:
        synthetic_data = privacy_engine.add_noise(synthetic_data, epsilon)

    # Compute utility metrics with epsilon for DP noise
    stats_similarity = compute_statistical_similarity(real_data, synthetic_data, epsilon)
    downstream_performance = compute_downstream_performance(real_data, synthetic_data, epsilon)

    return {
        'stats_similarity': stats_similarity,
        'downstream_performance': downstream_performance,
    }

def save_best_model(generator, metrics, best_metrics=None, best_model_state=None):
    """
    Save the best model based on evaluation metrics.
    """
    if best_metrics is None:
        best_metrics = {'stats_similarity': float('inf'), 'downstream_performance': 0}
        best_model_state = None

    if metrics['stats_similarity'] < best_metrics['stats_similarity']:
        best_metrics = metrics
        best_model_state = generator.state_dict()
        torch.save(best_model_state, 'best_model.pth')
        print(f"New best model saved with stats_similarity: {metrics['stats_similarity']}")

    return best_metrics, best_model_state
