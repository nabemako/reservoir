"""
Reservoir Computing Implementation using Echo State Networks (ESN)

This module implements a basic Reservoir Computer for time series prediction.
The reservoir uses a large recurrent network with fixed random weights, and only
the output layer is trained.

Reference: Based on Echo State Network (ESN) architecture
"""

import numpy as np
from typing import Tuple, Optional
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt


class ReservoirComputer:
    """
    Reservoir Computer using Echo State Network (ESN) approach.
    
    Architecture:
    - Input layer: connects to reservoir with random weights (W_in)
    - Reservoir: large recurrent network with fixed random weights (W_res)
    - Output layer: trained linear layer using Ridge regression (W_out)
    
    The reservoir state is updated as:
        r(n) = tanh(W_in * u(n) + W_res * r(n-1) + bias)
    
    The output is computed as:
        y(n) = W_out * r(n)
    """
    
    def __init__(
        self,
        input_size: int,
        reservoir_size: int = 300,
        spectral_radius: float = 0.9,
        sparsity: float = 0.9,
        input_scale: float = 0.5,
        ridge_alpha: float = 1e-6,
        random_state: Optional[int] = None
    ):
        """
        Initialize Reservoir Computer.
        
        Args:
            input_size: Number of input features
            reservoir_size: Number of neurons in the reservoir
            spectral_radius: Spectral radius of W_res (controls Lyapunov exponent)
            sparsity: Sparsity of reservoir connections (0-1)
            input_scale: Scaling factor for input weights
            ridge_alpha: Regularization parameter for Ridge regression
            random_state: Random seed for reproducibility
        """
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.input_scale = input_scale
        self.ridge_alpha = ridge_alpha
        
        if random_state is not None:
            np.random.seed(random_state)
        
        # Initialize weights
        self._initialize_weights()
        
        # Readout layer (will be trained)
        self.W_out = None
        self.trained = False
    
    def _initialize_weights(self):
        """Initialize reservoir and input weights."""
        # Input weights: uniform distribution [-1, 1] scaled by input_scale
        self.W_in = np.random.uniform(-1, 1, (self.reservoir_size, self.input_size))
        self.W_in *= self.input_scale
        
        # Reservoir weights: sparse random Gaussian matrix
        self.W_res = np.random.randn(self.reservoir_size, self.reservoir_size)
        
        # Apply sparsity constraint
        mask = np.random.rand(self.reservoir_size, self.reservoir_size) < self.sparsity
        self.W_res[mask] = 0
        
        # Scale to desired spectral radius
        eigenvalues = np.linalg.eigvals(self.W_res)
        rho = np.max(np.abs(eigenvalues))
        if rho > 0:
            self.W_res *= self.spectral_radius / rho
        
        # Bias term
        self.bias = np.random.uniform(-1, 1, self.reservoir_size)
    
    def _update_reservoir(
        self,
        x: np.ndarray,
        r: np.ndarray,
        activation: str = 'tanh'
    ) -> np.ndarray:
        """
        Update reservoir state based on input and previous state.
        
        Args:
            x: Input vector (input_size,)
            r: Current reservoir state (reservoir_size,)
            activation: Activation function ('tanh' or 'relu')
        
        Returns:
            Updated reservoir state (reservoir_size,)
        """
        raw = np.dot(self.W_in, x) + np.dot(self.W_res, r) + self.bias
        
        if activation == 'tanh':
            r_new = np.tanh(raw)
        elif activation == 'relu':
            r_new = np.maximum(0, raw)
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        return r_new
    
    def generate_states(
        self,
        X: np.ndarray,
        activation: str = 'tanh',
        warmup: int = 100
    ) -> np.ndarray:
        """
        Generate reservoir states for input sequence.
        
        Args:
            X: Input sequence (sequence_length, input_size)
            activation: Activation function ('tanh' or 'relu')
            warmup: Number of warmup steps to discard transients
        
        Returns:
            Reservoir states (sequence_length - warmup, reservoir_size)
        """
        sequence_length = X.shape[0]
        states = np.zeros((sequence_length, self.reservoir_size))
        r = np.zeros(self.reservoir_size)
        
        # Warmup phase: discard transient dynamics
        for t in range(warmup):
            r = self._update_reservoir(X[t], r, activation)
        
        # Recording phase: collect states
        for t in range(warmup, sequence_length):
            r = self._update_reservoir(X[t], r, activation)
            states[t] = r
        
        return states[warmup:]
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        activation: str = 'tanh',
        warmup: int = 100
    ) -> None:
        """
        Train the readout layer using Ridge regression.
        
        Args:
            X: Input sequences (sequence_length, input_size)
            y: Target outputs (sequence_length, output_size)
            activation: Activation function for reservoir
            warmup: Number of warmup steps
        """
        # Generate reservoir states
        states = self.generate_states(X, activation, warmup)
        
        # Adjust y to match the number of states (after warmup)
        y_train = y[warmup:]
        
        # Ensure same number of samples
        min_len = min(states.shape[0], y_train.shape[0])
        states = states[:min_len]
        y_train = y_train[:min_len]
        
        # Train output layer using Ridge regression
        ridge = Ridge(alpha=self.ridge_alpha)
        ridge.fit(states, y_train)
        
        self.W_out = ridge.coef_.T  # Shape: (reservoir_size, output_size)
        self.b_out = ridge.intercept_
        self.trained = True
    
    def predict(
        self,
        X: np.ndarray,
        activation: str = 'tanh',
        warmup: int = 100
    ) -> np.ndarray:
        """
        Make predictions using trained readout layer.
        
        Args:
            X: Input sequences (sequence_length, input_size)
            activation: Activation function
            warmup: Number of warmup steps
        
        Returns:
            Predictions (sequence_length - warmup, output_size)
        """
        if not self.trained:
            raise RuntimeError("Model must be trained before prediction")
        
        states = self.generate_states(X, activation, warmup)
        predictions = np.dot(states, self.W_out) + self.b_out
        
        return predictions


def create_nonlinear_timeseries_dataset(
    num_samples: int = 1000,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a nonlinear time series dataset using Mackey-Glass equation.
    
    This is a classic benchmark for time series prediction.
    
    Args:
        num_samples: Number of time steps
        seed: Random seed
    
    Returns:
        Tuple of (X, y) where X is input sequence and y is target
    """
    np.random.seed(seed)
    
    # Mackey-Glass time series
    x = np.zeros(num_samples)
    x[0:30] = 0.1
    
    for n in range(30, num_samples):
        x[n] = (0.9 * x[n-1] + 0.2 * x[n-30] / (1 + x[n-30] ** 10))
    
    # Create input-output pairs: use current value as input, next value as target
    X = x[:-1].reshape(-1, 1)
    y = x[1:].reshape(-1, 1)
    
    return X, y


def create_sin_cos_dataset(
    num_samples: int = 500,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a synthetic dataset combining sin and cos signals.
    
    Args:
        num_samples: Number of time steps
        seed: Random seed
    
    Returns:
        Tuple of (X, y)
    """
    np.random.seed(seed)
    t = np.linspace(0, 4 * np.pi, num_samples)
    
    # Input: sin(t)
    X = np.sin(t).reshape(-1, 1)
    
    # Target: sin(t) + 0.5 * cos(t) (nonlinear combination)
    y = (np.sin(t) + 0.5 * np.cos(t)).reshape(-1, 1)
    
    return X, y


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Evaluate model performance.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        Dictionary with evaluation metrics
    """
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)
    
    # Normalize by variance for normalized MSE
    var = np.var(y_true)
    nmse = mse / var if var > 0 else 0
    
    # Correlation coefficient
    corr = np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'NMSE': nmse,
        'Correlation': corr
    }


if __name__ == "__main__":
    print("=" * 70)
    print("Reservoir Computing (Echo State Networks) - Test Run")
    print("=" * 70)
    
    # Test 1: Mackey-Glass Time Series
    print("\n[Test 1] Mackey-Glass Time Series Prediction")
    print("-" * 70)
    
    X_mg, y_mg = create_nonlinear_timeseries_dataset(num_samples=800)
    split = 600
    
    X_train_mg = X_mg[:split]
    y_train_mg = y_mg[:split]
    X_test_mg = X_mg[split:]
    y_test_mg = y_mg[split:]
    
    print(f"Dataset shapes: X_train={X_train_mg.shape}, y_train={y_train_mg.shape}")
    print(f"                X_test={X_test_mg.shape}, y_test={y_test_mg.shape}")
    
    # Create and train RC model
    rc_mg = ReservoirComputer(
        input_size=1,
        reservoir_size=200,
        spectral_radius=0.95,
        sparsity=0.9,
        input_scale=0.5,
        ridge_alpha=1e-6,
        random_state=42
    )
    
    print("\nTraining Reservoir Computer...")
    rc_mg.train(X_train_mg, y_train_mg, warmup=50)
    
    # Make predictions
    print("Making predictions on test set...")
    pred_mg = rc_mg.predict(X_test_mg, warmup=50)
    
    # Evaluate
    metrics_mg = evaluate_model(y_test_mg[50:], pred_mg)
    print("\nPerformance Metrics:")
    for metric, value in metrics_mg.items():
        print(f"  {metric:15s}: {value:.6f}")
    
    # Test 2: Sin-Cos Dataset
    print("\n" + "=" * 70)
    print("[Test 2] Sin-Cos Time Series Prediction")
    print("-" * 70)
    
    X_sc, y_sc = create_sin_cos_dataset(num_samples=400)
    split_sc = 300
    
    X_train_sc = X_sc[:split_sc]
    y_train_sc = y_sc[:split_sc]
    X_test_sc = X_sc[split_sc:]
    y_test_sc = y_sc[split_sc:]
    
    print(f"Dataset shapes: X_train={X_train_sc.shape}, y_train={y_train_sc.shape}")
    print(f"                X_test={X_test_sc.shape}, y_test={y_test_sc.shape}")
    
    # Create and train RC model
    rc_sc = ReservoirComputer(
        input_size=1,
        reservoir_size=150,
        spectral_radius=0.9,
        sparsity=0.9,
        input_scale=0.5,
        ridge_alpha=1e-5,
        random_state=42
    )
    
    print("\nTraining Reservoir Computer...")
    rc_sc.train(X_train_sc, y_train_sc, warmup=30)
    
    # Make predictions
    print("Making predictions on test set...")
    pred_sc = rc_sc.predict(X_test_sc, warmup=30)
    
    # Evaluate
    metrics_sc = evaluate_model(y_test_sc[30:], pred_sc)
    print("\nPerformance Metrics:")
    for metric, value in metrics_sc.items():
        print(f"  {metric:15s}: {value:.6f}")
    
    # Visualization
    print("\n" + "=" * 70)
    print("Generating visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Mackey-Glass: training data
    ax = axes[0, 0]
    ax.plot(y_train_mg[:200], label='True', linewidth=2)
    ax.set_title('Test 1: Mackey-Glass Training Data')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Mackey-Glass: test predictions
    ax = axes[0, 1]
    test_range = range(min(100, len(y_test_mg[50:])))
    ax.plot(test_range, y_test_mg[50:50+len(test_range)], 'o-', label='True', linewidth=2, markersize=4)
    ax.plot(test_range, pred_mg[:len(test_range)], 's--', label='Predicted', linewidth=2, markersize=4)
    ax.set_title('Test 1: Mackey-Glass Test Predictions')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Sin-Cos: training data
    ax = axes[1, 0]
    ax.plot(y_train_sc[:150], label='True', linewidth=2)
    ax.set_title('Test 2: Sin-Cos Training Data')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Sin-Cos: test predictions
    ax = axes[1, 1]
    test_range = range(min(50, len(y_test_sc[30:])))
    ax.plot(test_range, y_test_sc[30:30+len(test_range)], 'o-', label='True', linewidth=2, markersize=4)
    ax.plot(test_range, pred_sc[:len(test_range)], 's--', label='Predicted', linewidth=2, markersize=4)
    ax.set_title('Test 2: Sin-Cos Test Predictions')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reservoir_computing_results.png', dpi=150, bbox_inches='tight')
    print("✓ Saved visualization to 'reservoir_computing_results.png'")
    
    print("\n" + "=" * 70)
    print("Test completed successfully!")
    print("=" * 70)
