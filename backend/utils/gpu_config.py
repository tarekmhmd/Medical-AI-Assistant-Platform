"""
GPU Configuration Module for Medical Assistant Platform
========================================================
Optimized for NVIDIA GTX 1650 (4GB VRAM)

Features:
- Automatic device selection (CUDA if available, else CPU)
- Mixed precision training (FP16)
- Memory optimization for low VRAM GPUs
- Optimal batch size calculation
"""

import torch
import gc

# =============================================================================
# GPU CONFIGURATION CLASS
# =============================================================================

class GPUConfig:
    """
    GPU configuration and utilities for deep learning training.
    Optimized for GTX 1650 (4GB VRAM).
    """
    
    def __init__(self):
        self.device = self._get_device()
        self.gpu_available = torch.cuda.is_available()
        self.mixed_precision = self.gpu_available
        
        if self.gpu_available:
            self.gpu_name = torch.cuda.get_device_name(0)
            self.gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            self.cuda_version = torch.version.cuda
            self.compute_capability = f"{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}"
            
            # Enable cuDNN benchmark for faster training
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
        else:
            self.gpu_name = "CPU"
            self.gpu_memory = 0
            self.cuda_version = None
            self.compute_capability = None
    
    def _get_device(self):
        """Automatically select best available device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def get_recommended_batch_size(self, model_type="cnn"):
        """
        Calculate recommended batch size based on GPU memory.
        GTX 1650 has 4GB VRAM.
        """
        if not self.gpu_available:
            return 4  # CPU default
        
        # For 4GB VRAM
        batch_sizes = {
            "cnn": 8,           # Simple CNN models
            "unet": 4,          # U-Net segmentation
            "transformer": 2,   # Transformers
            "audio": 16,        # Audio models (smaller inputs)
            "tabular": 32       # Tabular data
        }
        return batch_sizes.get(model_type, 4)
    
    def get_dataloader_config(self):
        """
        Get optimal DataLoader configuration for GPU.
        """
        if self.gpu_available:
            return {
                "pin_memory": True,
                "num_workers": 2,
                "persistent_workers": True
            }
        return {
            "pin_memory": False,
            "num_workers": 0,
            "persistent_workers": False
        }
    
    def get_scaler(self):
        """Get GradScaler for mixed precision training."""
        if self.mixed_precision:
            return torch.amp.GradScaler('cuda')
        return None
    
    def get_autocast_context(self):
        """Get autocast context manager for mixed precision."""
        if self.mixed_precision:
            return torch.amp.autocast('cuda')
        return None
    
    def clear_memory(self):
        """Clear GPU memory cache."""
        if self.gpu_available:
            torch.cuda.empty_cache()
            gc.collect()
    
    def get_memory_info(self):
        """Get current GPU memory usage."""
        if not self.gpu_available:
            return {"allocated": 0, "reserved": 0, "total": 0}
        
        return {
            "allocated": torch.cuda.memory_allocated(0) / (1024**2),
            "reserved": torch.cuda.memory_reserved(0) / (1024**2),
            "total": torch.cuda.get_device_properties(0).total_memory / (1024**2)
        }
    
    def to_device(self, model):
        """Move model to the configured device."""
        return model.to(self.device)
    
    def print_info(self):
        """Print GPU configuration info."""
        print("=" * 60)
        print("GPU CONFIGURATION")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"GPU Available: {self.gpu_available}")
        if self.gpu_available:
            print(f"GPU Name: {self.gpu_name}")
            print(f"GPU Memory: {self.gpu_memory:.2f} GB")
            print(f"CUDA Version: {self.cuda_version}")
            print(f"Compute Capability: {self.compute_capability}")
            print(f"Mixed Precision: {self.mixed_precision}")
            print(f"cuDNN Benchmark: Enabled")
        print("=" * 60)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_device():
    """Get the best available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_config():
    """Get GPU configuration instance."""
    return GPUConfig()


def prepare_model(model):
    """Prepare model for GPU training."""
    device = get_device()
    return model.to(device)


def prepare_batch(batch, device=None):
    """Move batch to device."""
    if device is None:
        device = get_device()
    
    if isinstance(batch, (list, tuple)):
        return [b.to(device) if torch.is_tensor(b) else b for b in batch]
    elif torch.is_tensor(batch):
        return batch.to(device)
    return batch


def train_step(model, batch, criterion, optimizer, scaler=None):
    """
    Perform one training step with mixed precision support.
    
    Args:
        model: Neural network model
        batch: Input batch (can be tuple of (inputs, labels))
        criterion: Loss function
        optimizer: Optimizer
        scaler: GradScaler for mixed precision (optional)
    
    Returns:
        loss: Training loss
    """
    device = get_device()
    
    if isinstance(batch, (list, tuple)):
        inputs, labels = batch[0].to(device), batch[1].to(device)
    else:
        inputs = batch.to(device)
        labels = None
    
    optimizer.zero_grad()
    
    if scaler is not None and torch.cuda.is_available():
        # Mixed precision training
        with torch.amp.autocast('cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, labels) if labels is not None else criterion(outputs)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        # Standard training
        outputs = model(inputs)
        loss = criterion(outputs, labels) if labels is not None else criterion(outputs)
        loss.backward()
        optimizer.step()
    
    return loss.item()


def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

# Create global config instance
gpu_config = GPUConfig()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    config = GPUConfig()
    config.print_info()
    
    print("\nRecommended batch sizes:")
    for model_type in ["cnn", "unet", "transformer", "audio", "tabular"]:
        print(f"  {model_type}: {config.get_recommended_batch_size(model_type)}")
    
    print("\nDataLoader config:")
    print(f"  {config.get_dataloader_config()}")
    
    print("\nMemory info:")
    mem = config.get_memory_info()
    print(f"  Allocated: {mem['allocated']:.1f} MB")
    print(f"  Reserved: {mem['reserved']:.1f} MB")
    print(f"  Total: {mem['total']:.1f} MB")
