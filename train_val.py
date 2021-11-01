import os
from os.path import join
from pathlib import Path
import ray
import ray.rllib.agents.ppo as ppo
from ray import tune
import wandb
from PIL import Image
from Modified_Dungeon import ModifiedDungeon
from tqdm import tqdm

SAVE_PATH = os.getcwd() + '/save/'
CHECKPOINT_PATH = os.getcwd() + '/ppo/'
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
if not os.path.exists(CHECKPOINT_PATH):
    os.makedirs(CHECKPOINT_PATH)
RUNS = 2
TRAIN = True


def logging_results(results, iteration):
    wandb.log({'iteration': iteration + 1,
              'episode_reward_min': results['episode_reward_min'],
              'episode_reward_mean': results['episode_reward_mean'],
              'episode_reward_max': results['episode_reward_max'],
              'episode_len_mean': results['episode_len_mean']})

seed = 10

config = ppo.DEFAULT_CONFIG.copy()
config['num_gpus'] = 1
config['log_level'] = 'INFO'
config['framework'] = 'torch'
config['env'] = 'ModifiedDungeon'
config['env_config'] = {
    'width': 20,
    'height': 20,
    'max_rooms': 3,
    'min_room_xy': 5,
    'max_room_xy': 10,
    'observation_size': 11,
    'vision_radius': 5,
    'max_steps': 1000
}
config['model'] = {
    'conv_filters': [
        [16, (3, 3), 2],
        [32, (3, 3), 2],
        [32, (3, 3), 1],
    ],
    'post_fcnet_hiddens': [32],
    'post_fcnet_activation': 'relu',
    'vf_share_layers': False,
}
config['rollout_fragment_length'] = 100
config['entropy_coeff'] = 0.1
config['lambda'] = 0.95
config['vf_loss_coeff'] = 1.0

def train(agent, save_path, iterations = 200):
    print('Start to train')
    checkpoint_dir = join(save_path, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    gif_dir = join(save_path, "gifs")
    if not os.path.exists(gif_dir):
        os.makedirs(gif_dir)

    for n in tqdm(range(iterations)):
        result = agent.train()
        file_name = agent.save(checkpoint_dir)
        logging_results(result, n)
        print(f'''\n 
        iter = {n + 1}, \n 
        episode_reward_min = {result["episode_reward_min"]}, \n
        episode_reward_mean =  {result['episode_reward_mean']}, \n
        'episode_reward_max'= {result['episode_reward_max']}, \n
        'episode_len_mean'= {result['episode_len_mean']}''')

        if (n + 1) % 5 == 0:
            artifact = wandb.Artifact("model", type="model")
            artifact.add_dir(Path(file_name).parent.absolute())
            wandb.log_artifact(artifact)

            env = ModifiedDungeon(20, 20, 3, min_room_xy=5, max_room_xy=10, vision_radius=5)
            env.seed(seed)
            env.action_space.seed(seed)

            obs = env.reset()

            frames = []

            for _ in range(400):
                action = agent.compute_single_action(obs)

                frame = Image.fromarray(env._map.render(env._agent)).convert('RGB').resize((500, 500), Image.NEAREST).quantize()
                frames.append(frame)

                obs, reward, done, info = env.step(action)
                if done:
                    break

            path_for_output = join(gif_dir, f'output_{n + 1}.gif')
            wandb.log({'gifs': wandb.Video(path_for_output, fps=30, format='gif')})
            frames[0].save(path_for_output, save_all=True, append_images=frames[1:], loop=0, duration=1000/60)

def val(agent, save_path, iters):
    env = ModifiedDungeon(20, 20, 3, min_room_xy=5, max_room_xy=10, vision_radius=5)
    obs = env.reset()

    for i in tqdm(iters):
        frames = []

        for _ in range(400):
            action = agent.compute_single_action(obs)

            frame = Image.fromarray(env._map.render(env._agent)).convert('RGB').resize((500, 500), Image.NEAREST).quantize()
            frames.append(frame)

            obs, reward, done, info = env.step(action)
            if done:
                break
        path_for_output = join(save_path, f'val_{n + 1}.gif')
        frames[0].save(path_for_output, save_all=True, append_images=frames[1:], loop=0, duration=1000/60)
        obs = env.reset()



if TRAIN:
    ray.shutdown()
    ray.init(ignore_reinit_error=True)
    wandb.init(project='ProdStory-Sensor', entity="yumvolkova", config=config)
    tune.register_env("ModifiedDungeon", lambda config: ModifiedDungeon(**config))

    agent = ppo.PPOTrainer(config)
    train(agent, SAVE_PATH)
else:
    ray.shutdown()
    ray.init(ignore_reinit_error=True)
    tune.register_env("ModifiedDungeon", lambda config: ModifiedDungeon(**config))
    agent = ppo.PPOTrainer(config)
    agent.restore(CHECKPOINT_PATH)
    val(agent, SAVE_PATH, RUNS)

