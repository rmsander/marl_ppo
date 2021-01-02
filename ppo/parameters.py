"""Contains hyperparameters and flags for the PPO MARL training (ppo_marl.py)."""


SIZE = (64, 64)  # Dimensions of the RGB observation window for agents
NUM_FRAMES = 1  #  Number of frames in the observation
NUM_CHANNELS = 1  # Number of color channels - 3 for RGB, 1 for grayscale
NORMALIZE = True  # Set pixel range from [0, 255] --> [0, 1]
H_RATIO = 0.25  # Height ratio used for experiments
NUM_AGENTS = 2  # The number of cars in the environment
USE_SEPARATE_AGENTS = True  # Whether to use N PPO agents
USE_SELF_PLAY = False  # Whether to use a single, master PPO agent
assert (int(USE_SEPARATE_AGENTS) + int(USE_SELF_PLAY) < 2)
USE_TENSORBOARD = True  # Whether to plot loss, returns, and trajectories to tb
ADD_TO_VIDEO = True  # Whether or not to create .avi videos from training/eval
USE_CLI = False  # Whether to use command-line arguments for hyperparameters
USE_EGO_COLOR = True  # Whether to use ego-car color for rendering observations
BACKWARDS_FLAG = True  # Render a flag whenever a car is driving backwards
DIRECTION = 'CCW'  # Default direction for OpenAI gym track
USE_RANDOM_DIRECTION = True  # Whether to use randomly-generated track direction
LR = 8e-5  # Learning rate for PPO agent
USE_LSTM = False  # Boolean for whether to use an LSTM for actor/value nets
NUM_EVAL_EPISODES = 5  # Number of episodes to evaluate each evaluation interval
EVAL_STEPS_PER_EPISODE = 1000  # Evaluation steps per evaluation episode
EVAL_INTERVAL = 100  # How many training episodes between each evaluation episode
TOTAL_EPOCHS = 20000  # Total training episodes
TOTAL_STEPS = 1e7  # Total of training steps to take from trajectories
COLLECT_STEPS_PER_EPISODE = 1000  # Steps per training episode
EXPERIMENT_NAME = "frame_stack_self_play_{}"  # Directory for models and logging
USE_XVFB = False  # If true, wraps main() in a virtual frame buffer
EPSILON = 0.0  # Probability of selecting greedy action
SAVE_INTERVAL = 500  # Save policies every SAVE_INTERVAL epochs
LOG_INTERVAL = 1  # Log metrics/results to tensorboard every LOG_INTERVAL epochs