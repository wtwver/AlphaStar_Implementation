import gym
import numpy as np
import tensorflow as tf
import glob

# Create the environment
env = gym.make("LunarLander-v2")

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)])

model = tf.keras.models.load_model('LunaLander_SL_Model')
save_path = 'Set Your Path'

episode_list, action_list, reward_list, done_list, step_list, state_list, next_state_list = [], [], [], [], [], [], []
for i_episode in range(0, 10000):
    observation = tf.constant(env.reset(), dtype=tf.float32)
    total_step, reward_sum = 0, 0
    while True:
        total_step += 1
        env.render()

        observation = tf.expand_dims(observation, 0)
        action_probs, _ = model(observation)
        action = np.argmax(np.squeeze(action_probs))

        observation_1, reward, done, info = env.step(action)
        observation_1 = tf.constant(observation_1, dtype=tf.float32)
        
        reward_sum += reward

        episode_list.append(i_episode)
        action_list.append(action)
        reward_list.append(reward)
        done_list.append(done)
        step_list.append(step)
        state_list.append(observation)
        next_state_list.append(observation_1)

        observation = observation_1
        if done:
            print("Total reward: {:.2f},  Total step: {:.2f}".format(reward_sum, total_step))
            total_step, reward_sum = 0, 0
            observation = env.reset()
            
            print("Save data")
            save_data = {'episode': i_episode, 'step': step_list, 
                         'state': state_list, 'next_state': next_state_list, 
                         'action': action_list, 'reward': reward_list, 'done': done_list}

            save_file = '/data_' + str(i_episode)
            path_npy = save_path + save_file + '.npy'
            np.save(path_npy, save_data)

            episode_list, action_list, reward_list, done_list, step_list, state_list, next_state_list = [], [], [], [], [], [], []
            
            break