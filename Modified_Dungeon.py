import os
import sys
from gym import spaces

sys.path.append(os.path.abspath("mapgen"))
os.environ["PYTHONPATH"] = os.path.abspath("mapgen")
from mapgen import Dungeon

class ModifiedDungeon(Dungeon):
    def __init__(self, width=20, height=20, max_rooms=3, min_room_xy=5, max_room_xy=12,
                 max_steps=1000, observation_size=11, vision_radius=5, seed = 10):
        super().__init__(
            width=width,
            height=height,
            max_rooms=max_rooms,
            min_room_xy=min_room_xy,
            max_room_xy=max_room_xy,
            observation_size=observation_size,
            vision_radius=vision_radius,
            max_steps=max_steps
        )

        self.seed(seed)
        self.observation_space = spaces.Box(0, 1, [observation_size, observation_size, 3])
        self.action_space = spaces.Discrete(3)

    def reset(self):
        observation = super().reset()
        return observation[:, :, :-1]

    def step(self, action):
        observation, reward, done, info = super().step(action)
        observation = observation[:, :, :-1]
        # reward = reward - (info['step'] / self._max_steps) + (info['new_explored'] / info['step'])

        '''
        info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
              - step: current step number
              - total_cells: total number of visible cells for current map
              - total_explored: total number of explored cells (map is solved when total_explored == total_cells)
              - new_explored: number of explored cells during this step
              - moved: whether an agent made a move (didn't collide with an obstacle)        
        '''
        if info['moved']:
            reward = 0.5 * reward + (info['total_explored'] / info['total_cells']) - (info['step'] / self._max_steps)
        else:
            reward = -1
        return observation, reward, done, info
