from pysc2.lib import features, point, actions, units
from pysc2.env.environment import TimeStep, StepType
from pysc2.env import sc2_env, available_actions_printer
from pysc2 import run_configs
from s2clientprotocol import sc2api_pb2 as sc_pb
from absl import flags
flags.FLAGS([__file__])
replay_file_path = '/home/a/AlphaStar_Implementation/replay/r/Simple64_2021-06-28-06-11-28.SC2Replay'

run_config = run_configs.get()
sc2_proc = run_config.start()
controller = sc2_proc.controller
ping = controller.ping()

replay_data = run_config.replay_data(replay_file_path)
info = controller.replay_info(replay_data)
print('info: ', info)

screen_size_px = (32, 32)
minimap_size_px = (32, 32)
player_id = 1
discount = 1.
step_mul = 1

screen_size_px = point.Point(*screen_size_px)
minimap_size_px = point.Point(*minimap_size_px)
interface = sc_pb.InterfaceOptions(raw=True, score=True, feature_layer=sc_pb.SpatialCameraSetup(width=24))
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
  print("error")
  exit()

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
print(_features)