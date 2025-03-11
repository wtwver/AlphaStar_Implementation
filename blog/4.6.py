def run_reinforcement_episode(
    initial_state: torch.Tensor,
    initial_reward_sum: torch.Tensor,
    model: torch.nn.Module,
    max_steps: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Runs a single episode to collect training data."""
    
    # Lists to collect data
    action_logits_list = []
    values_list = []
    rewards_list = []
    action_logits_ts_list = []
    sl_action_logits_ts_list = []
    
    state = initial_state
    reward_sum = initial_reward_sum
    
    # Set model to evaluation mode since we're collecting data
    model.eval()
    sl_model.eval()  # Assuming sl_model is defined elsewhere
    
    with torch.no_grad():
        for t in range(max_steps):
            # Add batch dimension
            state = state.unsqueeze(0)
            
            # Run the model
            action_logits_t, value = model(state)
            sl_action_logits_t, sl_value = sl_model(state)
            
            # Sample action from distribution
            dist = torch.distributions.Categorical(logits=action_logits_t[0])
            action = dist.sample()
            
            # Store values
            values_list.append(value.squeeze())
            action_logits_ts_list.append(action_logits_t[0])
            sl_action_logits_ts_list.append(sl_action_logits_t[0])
            action_logits_list.append(action_logits_t[0, action])
            
            # Apply action to environment
            state, reward, done = tf_env_step(action)  # Assuming this returns PyTorch tensors
            state = state.reshape(initial_state.shape)  # Maintain original shape
            
            reward_sum = reward_sum + reward.item()  # Convert to scalar for sum
            rewards_list.append(reward)
            
            if done.item():  # Convert to Python boolean
                break
    
    # Convert lists to tensors
    action_logits = torch.stack(action_logits_list)
    values = torch.stack(values_list)
    rewards = torch.stack(rewards_list)
    action_logits_ts = torch.stack(action_logits_ts_list)
    sl_action_logits_ts = torch.stack(sl_action_logits_ts_list)
    
    return action_logits, values, rewards, action_logits_ts, sl_action_logits_ts