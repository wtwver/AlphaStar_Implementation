import hickle as hkl 

self.home_feature_screen = []
self.home_feature_minimap = []
self.home_player = []
self.home_feature_units = []
self.home_game_loop = []
self.home_available_actions = []
self.home_build_queue = []
self.home_production_queue = []
self.home_single_select = []
self.home_multi_select = []
self.home_score_cumulative = []

self.home_action = []

for replay_step in range(0, _episode_length):
  controller.step(step_mul)
  obs = controller.observe()
  if obs.player_result: # Episode over.
    _state = StepType.LAST
    discount = 0
  else:
    discount = discount
    _episode_steps += step_mul

  agent_obs = _features.transform_obs(obs)
  if len(obs.actions) != 0:
     exec_actions = []
     for ac in obs.actions:
       exec_act = _features.reverse_action(ac)
      
       a_0 = int(exec_act.function)
       a_l = []
       for argument in exec_act.arguments:
         if str(type(argument[0])) != "<class 'int'>": 
	   a_l.append(argument[0].value)
	 else:
	   a_l.append(argument)
							
	 exec_actions.append([a_0, a_l])
	 self.home_action.append(exec_actions)
   else:
     exec_actions = []
     a_0 = 0
     a_l = [0]
     exec_actions.append([a_0, a_l])
     if replay_step % 8 == 0 or _state == StepType.LAST:
       self.home_action.append(exec_actions)
       pass
     else:
       continue
					
     done = 0
     if _state == StepType.LAST:
       done = 1
					
   self.home_feature_screen.append(agent_obs['feature_screen'])
   self.home_feature_minimap.append(agent_obs['feature_minimap'])
   self.home_player.append(agent_obs['player'])
   self.home_feature_units.append(agent_obs['feature_units'])
   self.home_game_loop.append(agent_obs['game_loop'])
   self.home_available_actions.append(agent_obs['available_actions'])
   self.home_build_queue.append(agent_obs['build_queue'])
   self.home_production_queue.append(agent_obs['production_queue'])
   self.home_single_select.append(agent_obs['single_select'])
   self.home_multi_select.append(agent_obs['multi_select'])
   self.home_score_cumulative.append(agent_obs['score_cumulative'])

   step = TimeStep(step_type=_state, reward=0,
		   discount=discount, observation=agent_obs)

   if _state == StepType.LAST:
     file_path = arguments.saving_path + replay_file_name + '.hkl'
     data = {'home_feature_screen': self.home_feature_screen, 
             'home_feature_minimap': self.home_feature_minimap, 
             'home_player': self.home_player,
             'home_feature_units': self.home_feature_units,
             'home_game_loop': self.home_game_loop,
             'home_available_actions': self.home_available_actions,
             'home_action': self.home_action,
             'home_build_queue': self.home_build_queue,
             'home_production_queue': self.home_production_queue,
             'home_single_select': self.home_single_select,
             'home_multi_select': self.home_multi_select,
             'home_score_cumulative': self.home_score_cumulative
	    }

	    self.home_feature_screen = []
	    self.home_feature_minimap = []
	    self.home_player = []
	    self.home_feature_units = []
	    self.home_game_loop = []
	    self.home_available_actions = []
	    self.home_build_queue = []
	    self.home_production_queue = []
	    self.home_single_select = []
	    self.home_multi_select = []
	    self.home_score_cumulative = []

	    self.home_action = []

	    hkl.dump(data, file_path)
	    break

  _state = StepType.MID