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


class BomberGame(gym.Env):
    # Setup the environment action and observation shapes
    def __init__(self):
        
        self.observation_space = Box(low=0, high=255, shape=(1,83,100), dtype=np.uint8)
        self.action_space = Discrete(3)
        self.cap = mss()
        
        self.game_location = {'top': 50, "left": 1275, 'width': 500, 'height': 450}
        self.done_location = {'top': 270, "left": 1435, 'width': 150, 'height': 25}
    # What is called to do something in the game
    def step(self, action):
        
        action_map = {
            0: 'right',
            1: 'down',
            2: 'up',
            3: 'left', 
            4: 'f', # bomb
            5: 'no_op'
        }
        
        if action != 5:
            pydirectinput.press(action_map[action])
            
        done, done_cap = self.get_done()
        
        new_observation = self.get_observation()
        
        pass
    # Visualize the game
    def render(self):
        pass
    # Restart the game
    def reset(self):
        pass
    # Gain game window
    def get_observation(self):
        
        raw = np.array(self.cap.grab(self.game_location))[:,:,:3].astype(np.uint8)
        
        gray = cv2.cvtColor(raw, cv2.COLOR_BGRA2GRAY)
        
        resized = cv2.resize(gray, (100,83))
        
        channel = np.reshape(resized, (1,83,100))
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
# plt.imshow(env.get_observation()[0], cmap='gray')
plt.imshow(env.get_done()[1], cmap='gray')
print(env.get_done()[0])
plt.show()
