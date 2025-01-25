from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Distribution, Gamma
from torch.distributions.utils import _standard_normal

class MultivariateStudentT(Distribution):
    """
    Multivariate Student-t distribution with location `loc`, scale matrix `scale_tril`,
    and degrees of freedom `df`.
    """
    def __init__(self, df, loc, scale_tril):
        self.df = df.unsqueeze(-1) if df.dim() == 0 else df  # Ensure df has batch dim
        self.loc = loc
        self.scale_tril = scale_tril
        batch_shape = loc.shape[:-1]  # Everything except last dimension
        event_shape = loc.shape[-1:]  # Last dimension
        super().__init__(batch_shape, event_shape)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        
        # Sample from Gamma(df/2, 1/2) to get chi-square samples
        gamma = Gamma(concentration=self.df.squeeze(-1)/2, rate=torch.ones_like(self.df.squeeze(-1))/2)
        chi2 = gamma.rsample(sample_shape)
        
        # Apply Student's t transformation
        z = eps * torch.rsqrt(chi2 / self.df.squeeze(-1)).unsqueeze(-1)
        
        # Transform to desired location and scale
        return self.loc + torch.matmul(self.scale_tril, z.unsqueeze(-1)).squeeze(-1)

    def log_prob(self, value):
        dim = self.loc.size(-1)
        diff = value - self.loc
        
        # Solve triangular system
        M = torch.triangular_solve(diff.unsqueeze(-1), self.scale_tril, upper=False)[0].squeeze(-1)
        quad = torch.sum(M ** 2, dim=-1)
        
        # Compute log determinant of scale matrix
        det = torch.diagonal(self.scale_tril, dim1=-2, dim2=-1).abs().log().sum(-1)
        
        df = self.df.squeeze(-1)
        
        return (torch.lgamma((df + dim) / 2.0) - 
                torch.lgamma(df / 2.0) - 
                dim/2.0 * torch.log(df * np.pi) - 
                det - 
                ((df + dim) / 2.0) * torch.log1p(quad / df))

class ProbabilisticForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.location = nn.Linear(hidden_dim, output_dim)
        self.scale = nn.Linear(hidden_dim, output_dim * output_dim)
        self.df = nn.Parameter(torch.ones(1))

    def forward(self, x):
        batch_size = x.size(0)
        lstm_out, _ = self.lstm(x)
        loc = self.location(lstm_out[:, -1])
        
        # Reshape scale_tril to ensure proper dimensions
        scale_raw = self.scale(lstm_out[:, -1])
        scale_tril = scale_raw.reshape(batch_size, loc.size(1), loc.size(1))
        scale_tril = torch.tril(scale_tril)
        
        # Add small positive diagonal for numerical stability
        diag_idx = torch.arange(loc.size(1), device=scale_tril.device)
        scale_tril[:, diag_idx, diag_idx] += 1e-6
        
        # Ensure df has proper batch dimension
        df = torch.exp(self.df).expand(batch_size) + 2  # df > 2 for finite covariance
        
        return loc, scale_tril, df

def mix_gaussian_kernel(X, Y):
    bandwidths = [0.1, 1, 10, 100]
    K = 0
    for b in bandwidths:
        dist = torch.cdist(X, Y)
        K += torch.exp(-dist / (2 * b * b))
    return K / len(bandwidths)

def mmd_loss(source, target):
    Kxx = mix_gaussian_kernel(source, source)
    Kyy = mix_gaussian_kernel(target, target)
    Kxy = mix_gaussian_kernel(source, target)
    return (Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()).sqrt()

class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, data, seq_length):
        self.data = torch.FloatTensor(data)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        return (self.data[idx:idx + self.seq_length],
                self.data[idx + self.seq_length])

def train_model(model, train_loader, num_epochs, learning_rate=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        total_loss = 0
        for x_batch, y_batch in train_loader:
            # Forward pass
            loc, scale_tril, df = model(x_batch)
            
            # Create StudentT distribution
            try:
                dist = MultivariateStudentT(df=df, loc=loc, scale_tril=scale_tril)
                
                # Negative log likelihood loss
                nll_loss = -dist.log_prob(y_batch).mean()
                
                # MMD loss between predicted and actual distributions
                mmd = mmd_loss(loc, y_batch)
                
                # Combined loss
                loss = nll_loss + 0.1 * mmd
                
                # Backward pass with gradient clipping
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            except RuntimeError as e:
                print(f"Error in batch: {e}")
                print(f"Shapes - df: {df.shape}, loc: {loc.shape}, scale_tril: {scale_tril.shape}")
                continue
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}, DF: {df[0].item():.2f}")

def forecast(model, initial_sequence, num_steps, num_samples=100):
    model.eval()
    device = next(model.parameters()).device
    initial_sequence = initial_sequence.to(device)
    
    with torch.no_grad():
        predictions = []
        prediction_intervals = []
        current_seq = initial_sequence.clone()
        
        for _ in range(num_steps):
            # Ensure current_seq has batch dimension
            model_input = current_seq.unsqueeze(0)  # Add batch dimension
            loc, scale_tril, df = model(model_input)
            dist = MultivariateStudentT(df=df, loc=loc, scale_tril=scale_tril)
            
            # Generate multiple samples for uncertainty estimation
            samples = torch.stack([dist.rsample() for _ in range(num_samples)])
            
            # Mean prediction - remove batch dimension
            pred_mean = samples.mean(0).squeeze(0)  # Shape: [n_features]
            predictions.append(pred_mean)
            
            # 95% prediction intervals
            pred_intervals = torch.quantile(samples, torch.tensor([0.025, 0.975], device=device), dim=0)
            prediction_intervals.append(pred_intervals)
            
            # Update sequence with mean prediction
            # current_seq shape: [seq_length, n_features]
            # pred_mean shape: [n_features]
            current_seq = torch.cat([current_seq[1:], pred_mean.unsqueeze(0)], dim=0)
            
    return (torch.stack(predictions).cpu(),
            torch.stack(prediction_intervals).cpu())

def plot_forecasts(initial_sequence, predictions, prediction_intervals, feature_names=None):
    """
    Plot the forecasts with uncertainty intervals for each feature.
    
    Args:
        initial_sequence: Historical data tensor [seq_length, n_features]
        predictions: Predicted values tensor [num_steps, n_features]
        prediction_intervals: Prediction intervals tensor [2, num_steps, n_features]
        feature_names: List of feature names (optional)
    """
    # Convert to numpy if tensors
    if torch.is_tensor(initial_sequence):
        initial_sequence = initial_sequence.cpu().numpy()
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    if torch.is_tensor(prediction_intervals):
        prediction_intervals = prediction_intervals.cpu().numpy()
    
    # Print shapes for debugging
    print(f"Initial sequence shape: {initial_sequence.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Prediction intervals shape: {prediction_intervals.shape}")
    
    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(initial_sequence.shape[1])]
    
    n_features = initial_sequence.shape[1]
    seq_length = initial_sequence.shape[0]
    forecast_length = predictions.shape[0]
    
    # Create time indices
    historical_idx = np.arange(seq_length)
    forecast_idx = np.arange(seq_length-1, seq_length + forecast_length - 1)
    
    # Set up the plot
    fig, axes = plt.subplots(n_features, 1, figsize=(12, 4*n_features))
    if n_features == 1:
        axes = [axes]
    
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e']  # Historical, Prediction, Intervals
    
    for i, (ax, name) in enumerate(zip(axes, feature_names)):
        # Plot historical data
        ax.plot(historical_idx, 
                initial_sequence[:, i],
                color=colors[0], 
                label='Historical', 
                linewidth=2,
                marker='o',
                markersize=4)
        
        # Plot predictions
        ax.plot(forecast_idx,
                predictions[:, i],
                color=colors[1],
                label='Forecast',
                linewidth=2,
                linestyle='--')
        plt.show()
        # Plot uncertainty intervals
        # lower_bound = prediction_intervals[0, :, i]  # Lower bound for feature i
        # upper_bound = prediction_intervals[1, :, i]  # Upper bound for feature i
        
        # ax.fill_between(forecast_idx,
        #                lower_bound,
        #                upper_bound,
        #                color=colors[2],
        #                alpha=0.2,
        #                label='95% Interval')
        
        # # Add vertical line separating historical and forecast
        # ax.axvline(x=seq_length-1, color='red', linestyle=':', label='Forecast Start')
        
        # Customize plot
        ax.set_title(f'{name} - Time Series Forecast', pad=20)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        
        # Add text box with statistics
        historical_mean = initial_sequence[:, i].mean()
        forecast_mean = predictions[:, i].mean()
        forecast_std = predictions[:, i].std()
        interval_width = np.mean(upper_bound - lower_bound)
        
        stats_text = (f'Historical mean: {historical_mean:.2f}\n'
                     f'Forecast mean: {forecast_mean:.2f}\n'
                     f'Forecast std: {forecast_std:.2f}\n'
                     f'Avg interval width: {interval_width:.2f}')
        
        ax.text(0.02, 0.98, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.legend()
    
    plt.tight_layout()
    return fig

# Example usage
if __name__ == "__main__":
    # Generate sample multivariate time series with heavy tails
    n_samples = 1000
    n_features = 3
    seq_length = 10
    
    # Create synthetic data with correlations and occasional extreme values
    np.random.seed(42)
    data = np.zeros((n_samples, n_features))
    t = np.linspace(0, 4*np.pi, n_samples)
    
    # Base signals
    data[:, 0] = np.sin(t)
    data[:, 1] = np.cos(t)
    
    # Add heavy-tailed noise using Student's t distribution
    df_noise = 3  # Low degrees of freedom for heavy tails
    noise = np.random.standard_t(df_noise, size=(n_samples, n_features))
    data += 0.1 * noise
    
    # Add correlation between features
    data[:, 2] = 0.5 * data[:, 0] + 0.5 * data[:, 1] + 0.1 * noise[:, 2]
    
    # Create dataset and loader
    dataset = TimeSeriesDataset(data, seq_length)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize and train model
    model = ProbabilisticForecaster(n_features, 64, n_features)
    train_model(model, train_loader, num_epochs=100)
    
    # Generate forecasts with uncertainty estimates
    initial_seq = torch.FloatTensor(data[-2*seq_length:-seq_length])
    predictions, prediction_intervals = forecast(model, initial_seq, num_steps=50)
    feature_names = ['Revenue', 'Cost', 'Margin']
    # plot_forecasts(initial_seq, predictions, prediction_intervals)
    plt.plot(np.arange(seq_length), 
                data[-seq_length:],
                color="blue", 
                label='Historical', 
                linewidth=2,
                marker='o',
                markersize=4)
    plt.plot(np.arange(seq_length), 
                predictions[:seq_length],
                color="red", 
                label='pred', 
                linewidth=2,
                marker='x',
                markersize=4)
    plt.show()
    print("\nFinal model degrees of freedom:", model.df.exp().item() + 2)
    print("Shape of predictions:", predictions.shape)
    
    print("Shape of prediction intervals:", prediction_intervals.shape)    
    
# plt.figure(figsize=(12, 6))
# for i in range(n_features):
#     plt.plot(data[:, i], label=f'Feature {i+1}')
# plt.title('Synthetic Multivariate Time Series Data')
# plt.xlabel('Time')
# plt.ylabel('Value')
# plt.legend()
# plt.show()    