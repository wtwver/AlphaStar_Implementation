from pysc2 import run_configs
from absl import flags
import os
flags.FLAGS([__file__])

path = '../../DI-Star/data/replays'
for root, dirs, files in os.walk(path):
    for f in sorted(files):
        try:
            replay_file_path = os.path.abspath(os.path.join(path, f))
            print('===',f)

            run_config = run_configs.get()
            sc2_proc = run_config.start()
            replay_data = run_config.replay_data(replay_file_path)
            controller = sc2_proc.controller
            info = controller.replay_info(replay_data)
            print(info)
        except Exception as e:
            print('failed', f , e)
            continue

ping = controller.ping()
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