import torch
import faiss
import numpy as np
import logging
import gc
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from typing import Tuple, Optional
from scipy.stats import mode
import torch.distributed as dist


def extract_features_preallocated(model, device, dataloader):

    dataset = dataloader.dataset
    total_samples = len(dataset)

    # Automatically detect model precision
    model_dtype = next(model.parameters()).dtype

    # 1. Probe feature dimensions (Done once)
    with torch.no_grad():
        dummy_data = next(iter(dataloader))[0]
        # Handle possible list wrapping
        print("detect feature: Start move data to GPU")
        dummy_input = (dummy_data[0] if isinstance(dummy_data, list) else dummy_data).to(device, model_dtype)
        dummy_feat = model([dummy_input])
        if isinstance(dummy_feat, (list, tuple)): dummy_feat = dummy_feat[0]
        feat_dim = dummy_feat.view(dummy_feat.size(0), -1).size(1)

    # Pre-allocate memory
    all_features = np.zeros((total_samples, feat_dim), dtype='float32')
    all_labels = np.zeros(total_samples, dtype='int32')
    print("pre allocated mem successfully")

    start_idx = 0
    if dist.is_initialized():
        dist.barrier()
    with torch.no_grad():
        for i, x in enumerate(dataloader):

            print("Start move input data to GPU")

            # Handle V-JEPA specific list input format: x[0][0] is data, x[1] is label
            x_input = x[0][0].to(device, model_dtype, non_blocking=True)
            y = x[1]

            feat = model([x_input])
            if isinstance(feat, (list, tuple)): feat = feat[0]

            # Flatten features: [B, L, D] -> [B, L*D]
            feat = feat.view(feat.size(0), -1)

            # Move to CPU in one batch
            batch_size = feat.size(0)
            end_idx = start_idx + batch_size

            print("Start move features to CPU")

            all_features[start_idx:end_idx] = feat.cpu().to(torch.float32).numpy().astype('float32')
            all_labels[start_idx:end_idx] = y.to(torch.float32).numpy().astype('int32')
            if dist.is_initialized():
                dist.barrier()

            start_idx = end_idx
            if (i + 1) % 5 == 0:
                logging.info(f"Feature extraction: {end_idx}/{total_samples}")
    if dist.is_initialized():
        dist.barrier()
    return all_features, all_labels


# ====== Mini-batch kNN Evaluation ======
def knn_eval(model, train_loader, val_loader, device, k=5):
    model.eval()

    # --- 1. Extract Training Features (Feature Bank) ---
    train_features, train_labels = extract_features_preallocated(model, device, train_loader)

    # Calculate feature norm: if norm approaches 0, the model has likely collapsed
    avg_norm = np.linalg.norm(train_features, axis=1).mean()
    faiss.normalize_L2(train_features)

    # Construct index (Prefer IndexFlatIP for cosine similarity)
    index = faiss.IndexFlatIP(train_features.shape[1])
    index.add(train_features)

    # --- 2. Extract Validation Features ---
    val_features, val_labels = extract_features_preallocated(model, device, val_loader)
    # If values change slightly but Acc remains constant:
    # Learning rate might be too high, causing the model to collapse into a tiny representation space after some epoch, losing discriminative power.
    print(f"DEBUG: First 5 elements of first feature: {val_features[0, :5]}")

    # Statistics (Calculate before normalization to reflect true representation strength)
    embed_var = np.var(val_features, axis=0).mean()
    embed_mean = np.mean(val_features).item()

    # Search after normalization
    faiss.normalize_L2(val_features)
    distances, indices = index.search(val_features, k=k)

    # --- 3. Fast Batch Voting (Replaces Python for-loop) ---
    neighbor_labels = train_labels[indices]  # [N_val, k]
    # Using scipy's mode function computes mode for all samples at once, 100x faster than loops
    knn_preds, _ = mode(neighbor_labels, axis=1, keepdims=False)

    acc = accuracy_score(val_labels, knn_preds)

    # Monitoring Alert
    if avg_norm < 1e-3:
        logging.warning(f"!!! ALERT: Feature Norm is too small ({avg_norm:.6f}). Model might be collapsing.")

    del train_features, val_features, index
    gc.collect()
    torch.cuda.empty_cache()  # Clear unused cached memory from GPU

    model.train()
    return acc, embed_var, embed_mean



