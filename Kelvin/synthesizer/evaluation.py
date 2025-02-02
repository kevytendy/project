import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def add_dp_noise(data, epsilon, sensitivity=1.0):
    """
    Add Laplace noise to the data for differential privacy.
    """
    if epsilon <= 0:
        raise ValueError("Epsilon must be positive for DP noise addition.")

    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale, data.shape)

    # Ensure numerical stability (clip values if necessary)
    noisy_data = data + noise
    return np.clip(noisy_data, 0, None)  # Clip negative values if needed


def compute_statistical_similarity(real_data, synthetic_data, epsilon):
    """
    Compute statistical similarity between real and synthetic data with DP noise.
    """
    if real_data.shape != synthetic_data.shape:
        raise ValueError("Shape mismatch: real_data and synthetic_data must have the same shape.")

    # Add DP noise to real data frequency distribution
    real_data_noisy = add_dp_noise(real_data, epsilon)

    # Compute similarity (L2 norm between mean vectors)
    real_mean = real_data_noisy.mean(axis=0)
    synthetic_mean = synthetic_data.mean(axis=0)
    similarity = np.linalg.norm(real_mean - synthetic_mean)

    return similarity


def compute_downstream_performance(real_data, synthetic_data, epsilon):
    """
    Train a downstream model on synthetic data and evaluate on real data with DP noise.
    """
    if real_data.shape[1] != synthetic_data.shape[1]:
        raise ValueError("Feature mismatch: real_data and synthetic_data must have the same number of columns.")

    # Add DP noise to real data
    real_data_noisy = add_dp_noise(real_data, epsilon)

    # Prepare data
    X_train, y_train = synthetic_data[:, :-1], synthetic_data[:, -1]
    X_test, y_test = real_data_noisy[:, :-1], real_data_noisy[:, -1]

    # Convert labels to integers if necessary
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    # Train and evaluate downstream model
    try:
        model = LogisticRegression(max_iter=200)  # Increased iterations for better convergence
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
    except Exception as e:
        print(f"Error in downstream model evaluation: {e}")
        accuracy = 0.0  # Return 0 accuracy if there's a failure

    return accuracy
