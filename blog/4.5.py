def run_supervised_episode(
    model: torch.nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
    """Runs a single episode to collect training data."""
    
    # Lists to collect data (will convert to tensors later)
    action_probs_list = []
    actions_list = []
    
    # File handling remains the same
    file_list = glob.glob("Expert data file path of your workspace")
    file = random.choice(file_list)
    
    data = np.load(file, allow_pickle=True)
    data = np.reshape(data, 1)
    data = data[0]
    
    state_list = data['state']
    action_list = data['action']
    reward_list = data['reward']
    done_list = data['done']
    next_state_list = data['next_state']
    
    reward_sum = 0
    # Set model to evaluation mode since we're just collecting data
    model.eval()
    with torch.no_grad():  # No gradient computation needed
        for t in range(0, len(data['state']) - 1):
            # Convert state to tensor and add batch dimension
            state = torch.from_numpy(state_list[t]).float()
            state = state.unsqueeze(0)  # Add batch dimension
            
            # Run the model
            action_logits_t, value = model(state)
            action_probs_t = torch.softmax(action_logits_t, dim=-1)
            
            # Store action (convert to tensor if not already)
            actions_list.append(torch.tensor(action_list[t]))
            
            reward_sum += reward_list[t]
            
            # Store action probabilities (remove batch dimension)
            action_probs_list.append(action_probs_t[0])
    
    # Stack the collected data into tensors
    action_probs = torch.stack(action_probs_list)
    actions = torch.stack(actions_list)
    
    return action_probs, actions