def supervised_replay(self, replay_feature_screen_list, replay_feature_minimap_list,
                      replay_feature_player_list, replay_feature_units_list,
                      replay_available_actions_list, replay_fn_id_list, replay_args_ids_list,
                      memory_state_list, carry_state_list,
                      replay_game_loop_list, last_action_type_list,
                      replay_build_queue_list, replay_single_select_list, replay_multi_select_list,
                      replay_score_cumulative_list):
    # Convert lists to tensors and concatenate
    replay_feature_screen_array = torch.cat(replay_feature_screen_list, dim=0)
    replay_feature_minimap_array = torch.cat(replay_feature_minimap_list, dim=0)
    replay_feature_player_array = torch.cat(replay_feature_player_list, dim=0)
    replay_feature_units_array = torch.cat(replay_feature_units_list, dim=0)
    replay_memory_state_array = torch.cat(memory_state_list, dim=0)
    replay_carry_state_array = torch.cat(carry_state_list, dim=0)
    replay_game_loop_array = torch.cat(replay_game_loop_list, dim=0)
    last_action_type_array = torch.cat(last_action_type_list, dim=0)
    replay_available_actions_array = torch.cat(replay_available_actions_list, dim=0)
    replay_fn_id_array = torch.cat(replay_fn_id_list, dim=0)
    replay_arg_ids_array = torch.cat(replay_args_ids_list, dim=0)
    
    replay_build_queue_array = torch.cat(replay_build_queue_list, dim=0)
    replay_single_select_array = torch.cat(replay_single_select_list, dim=0)
    replay_multi_select_array = torch.cat(replay_multi_select_list, dim=0)
    replay_score_cumulative_array = torch.cat(replay_score_cumulative_list, dim=0)
    
    # Set model to training mode
    self.ActorCritic.train()
    self.optimizer_sl.zero_grad()
    
    # Prepare input dictionary
    input_ = {
        'feature_screen': replay_feature_screen_array,
        'feature_minimap': replay_feature_minimap_array,
        'feature_player': replay_feature_player_array,
        'feature_units': replay_feature_units_array,
        'memory_state': replay_memory_state_array,
        'carry_state': replay_carry_state_array,
        'game_loop': replay_game_loop_array,
        'available_actions': replay_available_actions_array,
        'last_action_type': last_action_type_array,
        'build_queue': replay_build_queue_array,
        'single_select': replay_single_select_array,
        'multi_select': replay_multi_select_array,
        'score_cumulative': replay_score_cumulative_array
    }
    
    # Forward pass
    prediction = self.ActorCritic(input_)
    fn_pi = prediction['fn_out']
    arg_pis = prediction['args_out']
    next_memory_state = prediction['final_memory_state']
    next_carry_state = prediction['final_carry_state']
    
    batch_size = fn_pi.shape[0]
    
    # Compute function ID loss
    replay_fn_id_array_onehot = torch.nn.functional.one_hot(replay_fn_id_array, num_classes=573)
    replay_fn_id_array_onehot = replay_fn_id_array_onehot.reshape(batch_size, 573)
    replay_fn_id_array_onehot = replay_fn_id_array_onehot * replay_available_actions_array
    
    cce = torch.nn.CrossEntropyLoss()
    fn_id_loss = cce(fn_pi, replay_fn_id_array)  # Note: CrossEntropyLoss expects class indices, not one-hot
    
    # Compute argument IDs loss
    arg_ids_loss = 0
    for index, arg_type in enumerate(actions.TYPES):
        replay_arg_id = replay_arg_ids_array[:, index]
        arg_pi = arg_pis[arg_type]
        
        arg_id_loss = cce(arg_pi, replay_arg_id)  # Directly use class indices
        arg_ids_loss += arg_id_loss
    
    # Compute regularization loss
    regularization_loss = torch.tensor(0.0, device=fn_pi.device)
    for param in self.ActorCritic.parameters():
        regularization_loss += torch.norm(param)  # L2 regularization
    
    total_loss = fn_id_loss + arg_ids_loss + 1e-5 * regularization_loss
    
    total_loss.backward()
    self.optimizer_sl.step()