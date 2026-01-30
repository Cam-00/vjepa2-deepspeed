import torch


def get_rehab_optimizer_params(model, rehab_lr, stable_lr, rehab_weight_decay, stable_weight_decay):
    """
    Configuration for rehabilitation training parameter groups specifically for Blocks 0-1.

    # Call before initializing DeepSpeed
    optimizer_params = get_rehab_optimizer_params(model, base_lr=1e-4, weight_decay=0.04)
    # Note: If using DeepSpeed and an optimizer is defined in ds_config,
    # you must remove the optimizer config from ds_config and pass it manually.
    optimizer = torch.optim.AdamW(optimizer_params)

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=ds_config_path,
        # ... other parameters
    )
    """
    # Define the list of damaged and stable layers
    damaged_layers = ["blocks.0."]

    # Prepare three types of parameter groups:
    # 1. Blocks 0-1: High LR, Zero Weight Decay (Force recovery/re-calibration)
    # 2. Other layers (with Decay): Normal LR, Normal Weight Decay
    # 3. Other layers (no Decay): e.g., LayerNorm or Bias
    rehab_params = []

    # Logic assumes model.named_parameters() retrieves all weights
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_damaged = any(layer in name for layer in damaged_layers)

        # Determine if parameters should exclude Weight Decay (e.g., bias or norm)
        no_decay = any(nd in name for nd in ["bias"])

        if is_damaged and not no_decay:
            # Rehab zone: Increase learning rate (2x~5x), disable WD
            rehab_params.append({
                "params": [param],
                "lr": rehab_lr,
                "weight_decay": rehab_weight_decay,
                "name": f"rehab_{name}"
            })
        elif no_decay:
            # Standard no-decay zone
            rehab_params.append({
                "params": [param],
                "lr": stable_lr,
                "weight_decay": 0.0,
                "name": f"no_decay_{name}"
            })
        else:
            # Stable healthy zone
            rehab_params.append({
                "params": [param],
                "lr": stable_lr,
                "weight_decay": stable_weight_decay,
                "name": f"default_{name}"
            })

    return rehab_params


# Layer-wise Learning Rate and Weight Decay configuration
def get_param_groups(
    model,
    base_lr_encoder=2.05e-5,
    base_lr_predictor=2.05e-4,
    base_wd=0.02,
):
    param_groups = []
    num_encoder_layers = len(model.encoder.backbone.blocks)
    num_predictor_layers = len(model.predictor.backbone.predictor_blocks)

    def no_weight_decay(param_name, param):
        param_name = param_name.lower()
        return "bias" in param_name or param.ndim <= 1

    # Encoder Embedding
    for name, param in model.encoder.backbone.patch_embed.named_parameters():
        if not param.requires_grad:
            continue
        if no_weight_decay(name, param):
            wd = 0.0
        else:
            wd = base_wd * 0.25
        param_groups.append({
            "params": [param],
            "lr": base_lr_encoder * 1.0,
            "weight_decay": wd,
            "name": f"encoder.patch_embed.{name}"
        })

    # Encoder transformer blocks
    for i, block in enumerate(model.encoder.backbone.blocks):
        if i < (num_encoder_layers / 3):
            lr_scale = 1.0
            wd_scale = 0.35
        elif (num_encoder_layers / 3) <= i < (num_encoder_layers / 3 * 2):
            lr_scale = 1.5
            wd_scale = 0.5
        else:
            lr_scale = 2.0
            wd_scale = 1.0
        for name, param in block.named_parameters():
            if not param.requires_grad:
                continue
            if no_weight_decay(name, param):
                wd = 0.0
            else:
                wd = base_wd * wd_scale
            param_groups.append({
                "params": [param],
                "lr": base_lr_encoder * lr_scale,
                "weight_decay": wd,
                "name": f"encoder.block{i}.{name}"
            })

    # Predictor predictor_embed
    for name, param in model.predictor.backbone.predictor_embed.named_parameters():
        if not param.requires_grad:
            continue
        if no_weight_decay(name, param):
            wd = 0.0
        else:
            wd = base_wd * 0.25
        param_groups.append({
            "params": [param],
            "lr": base_lr_predictor,
            "weight_decay": wd,
            "name": f"predictor.predictor_embed.{name}"
        })

    # Predictor transformer blocks
    for i, block in enumerate(model.predictor.backbone.predictor_blocks):
        if i < (num_predictor_layers / 3):
            lr_scale = 1.0
            wd_scale = 0.35
        elif (num_predictor_layers / 3) <= i < (num_predictor_layers / 3 * 2):
            lr_scale = 1.5
            wd_scale = 0.5
        else:
            lr_scale = 2.0
            wd_scale = 1.0
        for name, param in block.named_parameters():
            if not param.requires_grad:
                continue
            if no_weight_decay(name, param):
                wd = 0.0
            else:
                wd = base_wd * wd_scale
            param_groups.append({
                "params": [param],
                "lr": base_lr_predictor * lr_scale,
                "weight_decay": wd,
                "name": f"predictor.predictor_block{i}.{name}"
            })

    return param_groups


def update_optimizer_groups(optimizer, epoch, base_lr, default_wd):
    """
    Dynamically adjust rehabilitation strategy for Blocks 0-1 based on current Epoch.

    # --- Training Loop Example ---
    # 1. Initialize using get_rehab_optimizer_params with group names
    params = get_rehab_optimizer_params(model, base_lr=1e-4, weight_decay=0.04)
    optimizer = torch.optim.AdamW(params)

    # 2. Initialize DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model, optimizer=optimizer, config=ds_config_path
    )

    # 3. Main training loop
    for epoch in range(start_epoch, total_epochs):
        # [Key Point] Call the update function before each epoch begins
        update_optimizer_groups(optimizer, epoch, base_lr=1e-4, default_wd=0.04)

        for step, batch in enumerate(train_loader):
            # Standard training flow
            loss = model_engine(batch)
            model_engine.backward(loss)
            model_engine.step()
    """
    damaged_layers = ["blocks.0.", "blocks.1."]

    # Phase Determination
    if epoch < 5:
        # Phase 1: Rapid Recovery
        rehab_lr_factor = 2.0
        rehab_wd = 0.0
        stage_name = "Phase 1: Recovery (No WD, High LR)"
    elif 5 <= epoch < 15:
        # Phase 2: Stabilization
        rehab_lr_factor = 1.0
        rehab_wd = 0.01
        stage_name = "Phase 2: Stabilization (Low WD, Normal LR)"
    else:
        # Phase 3: Full Integration
        rehab_lr_factor = 1.0
        rehab_wd = default_wd
        stage_name = "Phase 3: Full Return (Normal WD & LR)"

    print(f"\n>>> Epoch {epoch}: Switching to {stage_name}")

    # Iterate through DeepSpeed optimizer parameter groups and modify
    for group in optimizer.param_groups:
        group_name = group.get('name', '')

        # Check if the group belongs to rehab blocks (0-1)
        if "rehab" in group_name:
            group['lr'] = base_lr * rehab_lr_factor
            group['weight_decay'] = rehab_wd
            print(f"    Updated {group_name}: LR={group['lr']}, WD={group['weight_decay']}")
        else:
            # Stable layers maintain baseline configuration
            group['lr'] = base_lr
            group['weight_decay'] = group.get('weight_decay', default_wd)