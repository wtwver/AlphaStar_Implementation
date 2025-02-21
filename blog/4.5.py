def run_supervised_episode(
    model: tf.keras.Model) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  """Runs a single episode to collect training data."""

  action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  actions = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

  #file_list = glob.glob("/media/kimbring2/Steam/Supervised_A2C/expert_data/*.npy")
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
  for t in range(0, len(data['state']) - 1):
    # Convert state into a batched tensor (batch size = 1)
    state = state_list[t]
  
    # Run the model and to get action probabilities and critic value
    action_logits_t, value = model(state)
    action_probs_t = tf.nn.softmax(action_logits_t)
    actions = actions.write(t, action_list[t])
    
    reward_sum += reward_list[t]
    
    # Store log probability of the action chosen
    action_probs = action_probs.write(t, action_probs_t[0])
  
  action_probs = action_probs.stack()
  actions = actions.stack()

  return action_probs, actions