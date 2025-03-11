supervised_optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

def train_supervised_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer) -> torch.Tensor:
    """Runs a model training step."""
    
    # Set model to training mode
    model.train()
    
    # Zero the gradients
    optimizer.zero_grad()
    
    # Run the model for one episode to collect training data
    action_probs, actions = run_supervised_episode(model)
    
    # Convert training data to appropriate tensor shapes
    action_probs = action_probs.unsqueeze(1)  # Adds dimension at index 1
    actions = actions.unsqueeze(1)
    
    # Calculate loss
    loss = compute_supervised_loss(action_probs, actions)
    
    # Add regularization loss if any (from model.weight_decay or custom regularization)
    regularization_loss = torch.tensor(0.0, device=action_probs.device)
    for param in model.parameters():
        if param.requires_grad:
            regularization_loss += torch.norm(param)  # L2 regularization example
    
    total_loss = loss + regularization_loss
    
    # Backward pass: compute gradients
    total_loss.backward()
    
    # Apply gradients
    optimizer.step()
    
    return total_loss
