import torch
import cv2
import matplotlib.pyplot as plt
import math
import ctypes
import pickle
from copy import deepcopy
import os

def find_all_zero_subtensor(tensor):
    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            if torch.all(tensor[i, j, :] == 0):
                return (i, j)
    return None


def find_false_value(tensor):
    indices = torch.where(tensor == False)
    if len(indices[0]) > 0:
        return indices[0][0].item(), indices[1][0].item(), indices[2][0].item()
    else:
        return None

def get_images_matches_distance(img1, img2):
    # Initialize SIFT detector
    if isinstance(img1, torch.Tensor):
        img1 = img1.numpy()
        img2 = img2.numpy()

    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Initialize BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Match descriptors
    matches = bf.match(des1, des2)

    # Sort matches by distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)

    avg_distance = sum([m.distance for m in matches]) / len(matches)

    return avg_distance


def show_batch_images(images):
    if isinstance(images, torch.Tensor):
        images = images.numpy()

    # Create a figure with subplots
    size = int(math.sqrt(len(images)))
    fig, axes = plt.subplots(size, len(images)//size+1, figsize=(15, 15))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Iterate through each image and plot it
    for i in range(len(images)):
        img = images[i]
        axes[i].imshow(img)
        axes[i].axis('off')  # Hide the axis

    # Remove the unused subplot

    # Display the plot
    plt.tight_layout()
    plt.show()


class StateManager:
    def __init__(self, env):
        self.base_env = env
        self.states = []
        self.states.append(deepcopy(env.get_state()))

    def append(self, env):
        self.states.append(deepcopy(env.get_state()))

    def reset_to_step(self, index=0):
        need_state = self.states[index]
        del self.states[index+1:]

        return need_state

    def reverse_play_envs(self):
        states = self.states
        frames = []
        env = self.base_env

        for i in range(len(states)):
            print('reset to {}'.format(-(1+i)))
            env.reset_to(states[-(1+i)])
            frame = env.render(mode='rgb_array', height=128, width=128)

            frames.append(frame)

        dir = 'tmp_video'

        if not os.path.exists(dir):
            os.mkdir(dir)

        for i, f in enumerate(frames):
            cv2.imwrite(dir + '/' + str(i) + '.jpg', f)





