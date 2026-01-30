import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict


class GradientMonitor:
    def __init__(self, model, writer=None, layer_types=(torch.nn.Linear, torch.nn.Conv2d)):
        """
        Initialize the Gradient Monitor.
        Args:
            model: The PyTorch model to monitor.
            writer: TensorBoard SummaryWriter object (optional).
            layer_types: Layer types to monitor (default: Linear and Conv2d).
        """
        self.model = model
        self.writer = writer
        self.gradient_stats = defaultdict(list)              # Store gradient statistics
        self.weight_stats = defaultdict(list)                # Store weight statistics
        self.grad_weight_ratio_stats = defaultdict(list)     # Store gradient/weight ratios
        self.layer_types = layer_types
        self._register_hooks()

    def _register_hooks(self):
        """Register backward hooks for specified layer types."""
        for name, layer in self.model.named_modules():
            if name == "":
                continue
            if isinstance(layer, self.layer_types):
                # Bind layer_name explicitly to avoid closure issues
                layer.register_full_backward_hook(self._create_hook(layer, name))

    def _create_hook(self, layer, layer_name):
        """ Create hook function to record gradient/weight statistics and their L2 norm ratios. """
        def hook(module, grad_input, grad_output):
            # Check if grad_output is valid and contains a tensor
            if grad_output is not None and len(grad_output) > 0 and grad_output[0] is not None:
                grad = grad_output[0]  # Get the output gradient of this layer
                try:
                    stats = {
                        f"{layer_name}/grad_norm": grad.norm(p=2, dim=-1).mean().item(),
                    }
                    for key, val in stats.items():
                        self.gradient_stats[key].append(val)
                    self._record_weight_stats(layer_name, layer)
                    self._grad_weight_ratio(layer_name)
                    # # Adaptive Gradient Clipping
                    # self.adaptive_gradient_clipping()
                except RuntimeError as e:
                    # Log a warning if gradient computation fails (e.g., NaN or inf values)
                    print(f"Warning: Could not compute stats for {layer_name}: {e}")
            else:
                print(f"No valid gradient for layer: {layer_name}, grad_output: {grad_output}")
        return hook

    def _record_weight_stats(self, layer_name, layer):
        """Record weight statistics for all weight parameters in the layer."""
        for param_name, param in layer.named_parameters():
            if 'weight' in param_name:  # Monitor weights only (ignore bias)
                # L2 norm of all elements after flattening the parameter
                self.weight_stats[f"{layer_name}/weight_norm"].append(param.norm().item())

    def _grad_weight_ratio(self, layer_name):
        """
        Monitor the ratio of gradient norm to weight norm for each layer.
        This ratio helps detect gradient explosion or vanishing issues.

        Args:
            layer_name (str): Name of the layer to monitor.
        """
        # Define keys to look up
        grad_key = f"{layer_name}/grad_norm"
        weight_key = f"{layer_name}/weight_norm"
        ratio_key = f"{layer_name}/grad_weight_ratio_norm"

        # Check if necessary statistics exist
        if grad_key not in self.gradient_stats:
            print(f"Warning: Gradient stats not found for {grad_key}")
            return

        if weight_key not in self.weight_stats:
            print(f"Warning: Weight stats not found for {weight_key}")
            return

        # Ensure statistics lists are not empty
        if not self.gradient_stats[grad_key] or not self.weight_stats[weight_key]:
            print(f"Warning: Empty stats for {layer_name}")
            return

        try:
            # Get the latest values
            grad_norm = self.gradient_stats[grad_key][-1]
            weight_norm = self.weight_stats[weight_key][-1]

            # Calculate ratio (add epsilon to avoid division by zero)
            ratio = grad_norm / (weight_norm + 1e-8)

            # Initialize ratio statistics list if it doesn't exist
            if ratio_key not in self.grad_weight_ratio_stats:
                self.grad_weight_ratio_stats[ratio_key] = []

            # Store the ratio
            self.grad_weight_ratio_stats[ratio_key].append(ratio)

        except ZeroDivisionError:
            print(f"Error: Division by zero for {layer_name}")
        except Exception as e:
            print(f"Error computing grad/weight ratio for {layer_name}: {e}")

    def log_to_tensorboard(self, global_step):
        """
        Write gradient statistics to TensorBoard.
        Args:
            global_step: Global step count (integer).
        """
        if not isinstance(global_step, int):
            raise ValueError("global_step must be an integer")

        if self.writer is not None:
            for key, values in self.gradient_stats.items():
                if values:  # Only log if there are values to avoid empty mean errors
                    self.writer.add_scalar(f"gradients/{key}", torch.tensor(values).mean().item(), global_step)

            for key, values in self.weight_stats.items():
                if values:
                    self.writer.add_scalar(f"weights/{key}", torch.tensor(values).mean().item(), global_step)

            for key, values in self.grad_weight_ratio_stats.items():
                if values:
                    self.writer.add_scalar(f"grad_weight_ratio/{key}", torch.tensor(values).mean().item(), global_step)

            self.gradient_stats.clear()  # Clear current statistics
            self.weight_stats.clear()
            self.grad_weight_ratio_stats.clear()
        else:
            print("Warning: No TensorBoard writer provided, skipping logging")

    def print_summary(self):
        """Print average gradient statistics for each layer."""
        if not self.gradient_stats:
            print("\n=== Gradient Summary: No statistics available ===")
            return

        print("\n ========== Gradient Summary ===========================================================")
        for key, values in self.gradient_stats.items():
            if values:  # Avoid computing on empty lists
                if "grad_norm" in key:  # Focus on gradient norm
                    values = torch.tensor(values)
                    min_val = values.min().item()
                    mean_val = values.mean().item()
                    max_val = values.max().item()
                    # Use ljust(70) for alignment and tabs for readability
                    print(f"{key.ljust(70)}: min = {min_val:.6f} \t mean= {mean_val:.6f} \t max = {max_val:.6f}")
            else:
                print(f"{key.ljust(70)}: No gradient statistics collected")
        print("=" * 150)

    @torch.no_grad()
    def adaptive_gradient_clipping(self, clip_factor=0.01, eps=1e-3):
        """
        Adaptive Gradient Clipping (AGC)

        Args:
            clip_factor: Clipping factor controlling intensity (typically 0.01-0.05).
            eps: Small constant to prevent division by zero and numerical instability.
        """
        for p in self.model.parameters():
            if p.grad is None:
                continue

            # Only clip parameters with weights (exclude bias/1D params)
            if p.dim() <= 1:
                continue

            # Compute norms
            param_norm = p.data.norm(2)  # L2 Norm
            grad_norm = p.grad.data.norm(2)

            # Calculate maximum allowed gradient norm
            max_allowed_norm = (param_norm + eps) * clip_factor

            # Apply clipping if necessary
            if grad_norm > max_allowed_norm:
                # Safe division to avoid numerical issues
                scale_factor = max_allowed_norm / (grad_norm + 1e-6)
                p.grad.data.mul_(scale_factor)