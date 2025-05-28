from mss import mss
import pydirectinput 
import cv2
import numpy as np
import pytesseract
from matplotlib import pyplot as plt
import time
import gymnasium as gym
import os
from gymnasium.spaces import Box, Discrete
from image_to_tile_matrix import image_to_tile_matrix
class BomberGame(gym.Env):
    # Setup the environment action and observation shapes
    def __init__(self):
        
        self.observation_space = Box(low=0, high=255, shape=(1,130,110), dtype=np.uint8)
        self.action_space = Discrete(6)
        self.cap = mss()
        
        self.game_location = {'top': 131, "left": 1325, 'width': 416, 'height': 352}
        self.done_location = {'top': 270, "left": 1435, 'width': 150, 'height': 25}
        
    # What is called to do something in the game
    def step(self, action):
        
            
        action_map = {
            0: 'right',
            1: 'down',
            2: 'up',
            3: 'left',
            4: 'f',
            5: 'no_op'
        }

        if action != 5:
            key = action_map[action]
            print(f"Performing action: {key}")
            pydirectinput.keyDown(key)
            time.sleep(0.2)
            pydirectinput.keyUp(key)
                    


        done, done_cap = self.get_done()
        new_observation = self.get_observation()
        if done:
            print('Game over detected')
            
        
        
        return new_observation, done
    # Visualize the game
    def render(self):
        pass
    # Restart the game
    def reset(self):
        time.sleep(1)
        pydirectinput.keyDown('ctrl')
        pydirectinput.press('r')
        pydirectinput.keyUp('ctrl')
        time.sleep(1)
        pydirectinput.press('Enter')
        return self.get_observation()
    # Gain game window
    def get_observation(self):
        
        raw = np.array(self.cap.grab(self.game_location))[:,:,:3].astype(np.uint8)
        
        resized = cv2.resize(raw, (130,110))
        
        print(image_to_tile_matrix(resized))
        plt.imshow(resized, cmap='gray')
        plt.show()
        
        
        channel = np.reshape(resized, (1,130,110))
        return channel
    
    # Game over
    def get_done(self):
        
        done_cap = np.array(self.cap.grab(self.done_location))
        gray_frame = cv2.cvtColor(done_cap, cv2.COLOR_BGRA2GRAY)

        template = cv2.imread('screenshots/game_over.png', 0)
        
        result = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        done = max_val > 0.8
        
        return done, gray_frame
    
    
env = BomberGame()

env.get_done()
env.get_observation()


# for episode in range(10):
#     obs = env.reset()
#     done = False
    
#     while not done:
        
#         obs, done = env.step(env.action_space.sample())