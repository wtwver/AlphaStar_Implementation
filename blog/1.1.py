#!/usr/bin/env python
from pysc2 import run_configs
from absl import flags
flags.FLAGS([__file__])
replay_file_path = '/home/a/AlphaStar_Implementation/replay/Simple64_2021-06-28-06-11-28.SC2Replay'

run_config = run_configs.get()
sc2_proc = run_config.start()
controller = sc2_proc.controller
replay_data = run_config.replay_data(replay_file_path)

ping = controller.ping()
info = controller.replay_info(replay_data)

player0_race = info.player_info[0].player_info.race_actual
player0_mmr = info.player_info[0].player_mmr
player0_apm = info.player_info[0].player_apm
player0_result = info.player_info[0].player_result.result
print(player0_race, player0_mmr, player0_apm, player0_result)

player1_race = info.player_info[1].player_info.race_actual
player1_mmr = info.player_info[1].player_mmr
player1_apm = info.player_info[1].player_apm
player1_result = info.player_info[1].player_result.result
print(player1_race, player1_mmr, player1_apm, player1_result)