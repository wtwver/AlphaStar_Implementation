cce_loss = torch.nn.CrossEntropyLoss()

def compute_supervised_loss(
    action_probs: torch.Tensor,
    actions: torch.Tensor) -> torch.Tensor:
    """Computes the supervised loss."""
    
    # In PyTorch, CrossEntropyLoss combines log_softmax and NLLLoss,
    # so we don't need to explicitly create one-hot encodings
    action_loss = cce_loss(action_probs, actions)
    
    return action_loss