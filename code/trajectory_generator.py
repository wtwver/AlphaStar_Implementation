#!/usr/bin/env python
from pysc2.lib import features, point, actions, units
from pysc2.env.environment import TimeStep, StepType
from pysc2.env import sc2_env, available_actions_printer
from pysc2 import run_configs
from s2clientprotocol import sc2api_pb2 as sc_pb

import os
import argparse
import pickle 
import gzip

from absl import flags
FLAGS = flags.FLAGS
FLAGS(['trajectory.py'])

parser = argparse.ArgumentParser(description='Trajetory File Generation')
parser.add_argument('--replay_path', type=str, help='Path of replay file')
parser.add_argument('--player_1', type=str, default='Terran', help='Race of player 1')
parser.add_argument('--player_2', type=str, default='Terran', help='Race of player 2')
parser.add_argument('--mmr', type=int, default=2500, help='Threshold of mmr score ')
parser.add_argument('--saving_path', type=str, help='Path for saving proprocessed replay file')

arguments = parser.parse_args()

class Trajectory(object):
	def __init__(self, source, saving_path, home_race_name, away_race_name, replay_filter, filter_repeated_camera_moves=False):
		self.replay_path = os.path.expanduser(source) if source.startswith('~') else os.path.abspath(source)
		self.saving_path = os.path.expanduser(saving_path) if saving_path.startswith('~') else os.path.abspath(saving_path)

		self.home_race_name = home_race_name
		self.away_race_name = away_race_name
		self.replay_filter = replay_filter
		self.filter_repeated_camera_moves = filter_repeated_camera_moves

		self.home_BO = None

		self.home_observation = []
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

		self.home_action = []

		self.home_score_cumulative = []

		self.away_BU = None
		self.away_trajectory = []
		self.away_score_cumulative = None

		print(self.saving_path)
		saving_folder_existence = os.path.isdir(self.saving_path)
		assert saving_folder_existence, "saving folder is not existed"

	def get_BO(self, player):
		if player == 0:
			return self.home_BO
		else:
			return self.away_BU

	def generate_trajectory(self):

		function_dict = {}
		for _FUNCTION in actions._FUNCTIONS:
			function_dict[_FUNCTION.ability_id] = _FUNCTION.name

		race_list = ['Terran', 'Zerg', 'Protoss']

		"""How many agent steps the agent has been trained for."""
		run_config = run_configs.get()
		sc2_proc = run_config.start()
		controller = sc2_proc.controller

		replay_files = sorted([ f for f in os.listdir(self.replay_path) if f.endswith('.SC2Replay') ])
		print(replay_files)
		assert len(replay_files) != 0, "No replay file is found"


		for replay_file in replay_files:
			try: 
				saving_file =  arguments.saving_path + '/' + replay_file + '.pkl'
				if os.path.exists(saving_file):
					continue
				print(f'===processing {saving_file}')

				replay_data = run_config.replay_data(self.replay_path + '/' + replay_file)
				ping = controller.ping()
				info = controller.replay_info(replay_data)

				player0_race = info.player_info[0].player_info.race_actual
				player0_mmr = info.player_info[0].player_mmr
				player0_apm = info.player_info[0].player_apm
				player0_result = info.player_info[0].player_result.result
				
				home_race = race_list.index(self.home_race_name) + 1
				if (home_race == player0_race):
					pass
				else:
					print(f"===player0_race fail {home_race} is not {player0_race}")
					continue

				if (player0_mmr >= self.replay_filter):
					pass
				else:
					pass
				
				player1_race = info.player_info[1].player_info.race_actual
				player1_mmr = info.player_info[1].player_mmr
				player1_apm = info.player_info[1].player_apm
				player1_result = info.player_info[1].player_result.result

				away_race = race_list.index(self.away_race_name) + 1
				if (away_race == player1_race):
					pass
				else:
					print(f"===player1_race fail {away_race} is not {player1_race}")
					continue

				if (player1_mmr >= self.replay_filter):
					pass
				else:
					pass
				
				screen_size_px = (32, 32)
				minimap_size_px = (32, 32)
				player_id = 1
				discount = 1.
				step_mul = 1

				screen_size_px = point.Point(*screen_size_px)
				minimap_size_px = point.Point(*minimap_size_px)
				interface = sc_pb.InterfaceOptions(raw=True, score=True,
					feature_layer=sc_pb.SpatialCameraSetup(width=24))
				screen_size_px.assign_to(interface.feature_layer.resolution)
				minimap_size_px.assign_to(interface.feature_layer.minimap_resolution)

				map_data = None
				if info.local_map_path:
					map_data = run_config.map_data(info.local_map_path)

				_episode_length = info.game_duration_loops
				_episode_steps = 0

				controller.start_replay(sc_pb.RequestStartReplay(replay_data=replay_data, 
					map_data=map_data, options=interface,
					observed_player_id=player_id))

				_state = StepType.FIRST

				if (info.HasField("error") or
				                    info.base_build != ping.base_build or  # different game version
				                    info.game_duration_loops < 1000 or
				                    len(info.player_info) != 2):
					# Probably corrupt, or just not interesting.
					print("===error")
					continue

				feature_screen_size = 32
				feature_minimap_size = 32
				rgb_screen_size = None
				rgb_minimap_size = None
				action_space = None
				use_feature_units = True
				aif = sc2_env.parse_agent_interface_format(
					feature_screen=feature_screen_size,
					feature_minimap=feature_minimap_size,
					rgb_screen=rgb_screen_size,
					rgb_minimap=rgb_minimap_size,
					action_space=action_space,
					use_feature_units=use_feature_units)

				_features = features.features_from_game_info(controller.game_info(), agent_interface_format=aif)

				build_info = []
				build_name = []
				replay_step = 0

				print("===_episode_length: {}".format(_episode_length))
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
							#print("exec_act.arguments: {}".format(exec_act.arguments))
							for argument in exec_act.arguments:
								#print("argument: {}".format(argument))
								if str(type(argument[0])) != "<class 'int'>": 
									a_l.append(argument[0].value)
								else:
									a_l.append(argument)
							
							exec_actions.append([a_0, a_l])

							#print("a_0: {}".format(a_0))
							#print("a_l: {}".format(a_l))
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
					
					#print("agent_obs['score_cumulative']: ", agent_obs['score_cumulative'])
					#print("agent_obs['multi_select']: ", agent_obs['multi_select'])
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

						self.home_action = []

						self.home_score_cumulative = []

						with gzip.open(saving_file, 'wb') as f:
							pickle.dump(data, f)
							print("===pickle saved", f)

						break

					_state = StepType.MID

				#self.home_BO = build_info
				#self.away_BU = score_cumulative_dict
			except Exception as e:
				print(f"Error occurred at line: {e.__traceback__.tb_lineno}")
				


replay = Trajectory(arguments.replay_path, arguments.saving_path, arguments.player_1, arguments.player_2, arguments.mmr)
replay.generate_trajectory()
