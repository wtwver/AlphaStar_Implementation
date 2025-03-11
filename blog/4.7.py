def compute_kl_loss(
    action_logits_ts: torch.Tensor,
    sl_action_logits_ts: torch.Tensor) -> torch.Tensor:
    """Computes the KL divergence loss."""
    
    # Create categorical distributions from logits
    dist = torch.distributions.Categorical(logits=action_logits_ts)
    sl_dist = torch.distributions.Categorical(logits=sl_action_logits_ts)
    
    # Compute KL divergence
    kl_loss = torch.distributions.kl_divergence(dist, sl_dist)
    kl_loss = torch.mean(kl_loss)
    
    return kl_loss