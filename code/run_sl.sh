#uv venv --python=3.9
uv pip install torch tensorboard numpy==2.0 tqdm pysc2 protobuf==3.20.2

caffeinate -s python run_sl.py --workspace_path ~/AlphaStar_Implementation/ --training True --gpu_use True --learning_rate 0.0001 --replay_hkl_file_path ~/AlphaStar_Implementation/pkl/h/ --environment Simple64