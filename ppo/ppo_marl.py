""" Class object and functions for creating, training, and evaluating PPO agents
using the TensorFlow Agents API. """

# Native Python imports
import os
import argparse
from datetime import datetime
import pickle

# TensorFlow and tf-agents
from tf_agents.environments import gym_wrapper, tf_py_environment
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
import tf_agents.trajectories.time_step as ts
from tf_agents.utils import common
from tf_agents.policies.policy_saver import PolicySaver

# Other external packages
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Custom modules
from utils import ObservationWrapper, video_summary, make_agent
from envs.multi_car_racing import MultiCarRacing
from parameters import *

# Get GPUs
from tensorflow.python.client import device_lib

# If a virtual frame buffer is needed
from xvfbwrapper import Xvfb


class PPOTrainer:
    """ Class object for training and evaluating PPO agents.

    A PPO Deep Reinforcement Learning trainer object using TensorFlow Agents.
    Uses PPO agent objects with TensorFlow environments to train agent(s) to
    maximize their reward(s) in their environment(s).

    Arguments:
        ppo_agent (PPOAgent): A PPO agent used for learning in the
            environment env.
        train_env (tf env): A TensorFlow environment that the agent interacts
            with via the neural networks.  Used for creating training
            trajectories for the agent, and for optimizing its networks.
        eval_env (tf env): A TensorFlow environment that the agent interacts
            with via the neural networks.  Used for evaluating the performance
            of the agent.
        size (tuple): The observation width and height for the agent.  Defaults
            to (96, 96).
        normalize (bool):  Whether or not to normalize observations from
            [0, 255] --> [0, 1].
        num_frames (int): The number of frames to use for the agent(s)'s
            observations.  If num_frames > 1, frame stacking is used.
            Defaults to 1.
        num_channels (int):  The number of channels to use for observations.
        use_tensorboard (bool): Whether or not to run training and
            evaluation with tensorboard.  Defaults to True.
        add_to_video (bool): Whether or not to create videos of the agent's
            training and save them as videos.  Defaults to True.
        use_separate_agents (bool): Whether or not to train with N different PPO
            agents (where N is the number of cars).  Defaults to False.
        use_self_play (bool): Whether or not to train with one master
            PPO agent to control all cars.  Defaults to False.
        num_agents (int):  The number of cars used to train the agent(s) in
            this environment.  Defaults to 2.
        use_lstm (bool): Whether LSTM-based actor and critic neural networks
            are used for the PPO agent.  Defaults to False.
        experiment_name (str): A name for the experiment where policies and
            summary statistics will be stored in the local filesystem.
        collect_steps_per_iteration (int):  The number of time steps collected
            in each trajectory of training for the PPO agent(s).  Defaults to
            1000.
        total_epochs (int):  The number of episodes the PPO agent is trained
            for.  Defaults to 1000.
        total_steps (int):  The total number of training steps the PPO agent is
            trained for.  Defaults to 1e6.
        eval_steps_per_episode (int):  The maximum number of time steps in an
            evaluation episode.  Defaults to 1000.
        eval_interval (int):  The number of training episodes between each
            subsequent evaluation period of the PPO agent.  Defaults to 100.
        num_eval_episodes (int):  The number of evaluation episodes in each
            evaluation period.  Defaults to 5.
        epsilon (float):  Probability of selecting actions using a greedy policy
            for a given episode during training.  Nonzero values can be used to
            reduce domain adaptation when transferring from collect to eval
            policies.
        save_interval (int): The policy is saved every save_interval number of
            training epochs.  Defaults to 500.
        log_interval (int):  Metrics and results are logged to tensorboard
            (if enabled) every log_interval epochs.
    """
    def __init__(self, ppo_agents, train_env, eval_env, size=(96, 96),
                 normalize=True, num_frames=1, num_channels=3,
                 use_tensorboard=True, add_to_video=True,
                 use_separate_agents=False, use_self_play=False,
                 num_agents=2, use_lstm=False, experiment_name="",
                 collect_steps_per_episode=1000, total_epochs=1000,
                 total_steps=1e6, eval_steps_per_episode=1000,
                 eval_interval=100, num_eval_episodes=5, epsilon=0.0,
                 save_interval=500, log_interval=1):

        # Environment attributes
        self.train_env = train_env  # Environment for training
        self.eval_env = eval_env  # Environment for testing

        # Observation attributes
        self.size = size
        self.H, self.W = self.size[0], self.size[1]  # Observation width/height
        self.normalize = normalize
        self.num_frames = num_frames
        self.num_channels = num_channels

        # MARL
        self.use_separate_agents = use_separate_agents  # Use N PPO agents (one for each car)
        self.use_self_play = use_self_play  # Use one master PPO agent
        self.use_lstm = use_lstm  # Whether agents maintain an LSTM
        self.num_agents = num_agents  # The number of cars in this environment

        # Specifics of training
        self.max_buffer_size = collect_steps_per_episode  # Entire memory buffer
        self.collect_steps_per_episode = collect_steps_per_episode  # Fill buffer
        self.epochs = total_epochs  # Total number of episodes
        self.total_steps = total_steps  # Total training stips
        self.global_step = 0  # Global step count
        self.epsilon = epsilon  # Probability of using greedy policy

        print("Total steps: {}".format(self.total_steps))

        # Create N different PPO agents
        if use_separate_agents and self.num_agents > 1:
            self.agents = ppo_agents  # Use copies
            for agent in self.agents:
                agent.initialize()  # Initialize all copied agents
            self.actor_nets = [self.agents[i]._actor_net \
                               for i in range(self.num_agents)]
            self.value_nets = [self.agents[i]._value_net \
                               for i in range(self.num_agents)]
            self.eval_policies = [self.agents[i].policy \
                                  for i in range(self.num_agents)]
            self.collect_policies = [self.agents[i].collect_policy \
                                     for i in range(self.num_agents)]
            self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
                self.agents[0].collect_data_spec,
                batch_size=self.train_env.batch_size,
                max_length=self.max_buffer_size)  # Create shared replay buffer

        # Create a single PPO agent
        else:
            self.agent = ppo_agents
            self.actor_net = self.agent._actor_net
            self.value_net = self.agent._value_net
            self.eval_policy = self.agent.policy
            self.collect_policy = self.agent.collect_policy
            self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
                self.agent.collect_data_spec,
                batch_size=self.train_env.batch_size,
                max_length=self.max_buffer_size)

        # Create observation wrapper(s)
        if self.num_agents > 1:  # Need N observation wrappers
            self.observation_wrappers = \
                            [ObservationWrapper(size=self.size, normalize=self.normalize,
                                                num_channels=self.num_channels,
                                                num_frames=self.num_frames)
                             for i in range(self.num_agents)]

        else:  # Single observation wrapper for single car
            self.observation_wrapper = ObservationWrapper(size=self.size,
                                                          normalize=self.normalize,
                                                          num_channels=self.num_channels,
                                                          num_frames=self.num_frames)

        # Evaluation
        self.num_eval_episodes = num_eval_episodes  # num_episodes to evaluate

        # Track evaluation performance for all PPO agents
        if self.use_separate_agents:
            self.eval_returns = [[] for i in range(self.num_agents)]

        # Track evaluation performance over a single agent
        else:
            self.eval_returns = []

        self.eval_interval = eval_interval  # Evaluate every <x> epochs
        self.max_eval_episode_steps = eval_steps_per_episode  # Steps in episode

        # Logging
        self.time_ext = datetime.now().strftime("%Y%m%d-%H%M")
        self.log_interval = log_interval
        
        # For creating AVI videos
        self.video_train = []
        self.video_eval = []
        self.add_to_video = add_to_video
        self.FPS = 50  # Frames per second

        # Save training and evaluation policies
        self.policy_save_dir = os.path.join(os.path.split(__file__)[0], "models",
                                            experiment_name.format(self.time_ext))
        self.save_interval = save_interval
        if not os.path.exists(self.policy_save_dir):
            print("Directory {} does not exist;"
                  " creating it now".format(self.policy_save_dir))
            os.makedirs(self.policy_save_dir, exist_ok=True)

        if self.use_separate_agents:
            # Get train and evaluation policies
            self.train_savers = [PolicySaver(self.collect_policies[i],
                                             batch_size=None) for i in
                                 range(self.num_agents)]
            self.eval_savers = [PolicySaver(self.eval_policies[i],
                                            batch_size=None) for i in
                                range(self.num_agents)]

        else:
            # Get train and evaluation policy savers
            self.train_saver = PolicySaver(self.collect_policy, batch_size=None)
            self.eval_saver = PolicySaver(self.eval_policy, batch_size=None)

        # Tensorboard
        self.log_dir = os.path.join(os.path.split(__file__)[0], "logging",
                                    experiment_name.format(self.time_ext))
        self.tb_file_writer = tf.summary.create_file_writer(self.log_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
        self.use_tensorboard = use_tensorboard  # Use tensorboard for plotting

        # Set width and height
        self.size = size
        self.H, self.W = self.size

        # Total frames to use for tensorboard
        self.stacked_channels = self.num_channels * self.num_frames

        # For creating tensorboard gif videos
        if self.use_tensorboard:
            self.tb_gif_train = np.zeros((self.collect_steps_per_episode,
                                         self.num_agents, self.H, self.W,
                                          self.stacked_channels))
            self.tb_gif_eval = np.zeros((self.max_eval_episode_steps,
                                        self.num_agents, self.H, self.W,
                                         self.stacked_channels))
        # Devices
        local_device_protos = device_lib.list_local_devices()
        num_gpus = len([x.name for x in local_device_protos if
                        x.device_type == 'GPU'])
        self.use_gpu = num_gpus > 0

    def is_last(self, mode='train'):
        """ Check if a time step is the last time step in an episode/trajectory.

        Checks the current environment (selected according to mode) using the
        tf-agents API to see if the time step is considered 'last'
        (time_step.step_type = 2).

        Arguments:
            mode (str): Whether to check if the 'train' environment or 'eval'
                environment is currently on its last time step for an episode.
                Defaults to 'train' (the training environment).

        Returns:
            is_last (bool): A boolean for whether the time step is last for any
                of the cars.
        """
        # Check if on the last time step
        if mode == 'train':
            step_types = self.train_env.current_time_step().step_type.numpy()
        elif mode == 'eval':
            step_types = self.eval_env.current_time_step().step_type.numpy()

        # See if any time steps are 'last'
        is_last = bool(min(np.count_nonzero(step_types == 2), 1))
        return is_last

    def get_agent_timesteps(self, time_step, step=0,
                            only_ego_car=False, ego_car_index=0, max_steps=1000):
        """ Create time step(s) for agent(s) that can be used with tf-agents.

        Helper function for computing the time step for a specific agent.
        Takes the action and observation corresponding to the input car index,
        and creates a time step using these reward and observation slices.  When
        adding trajectories to the replay buffer, this ensures that the data
        added is only with one car at a time.

        This function is used as a wrapper for mapping time steps to a format
        that can be used for action selection by agents.

        Arguments:
            time_step (tf-agents TimeStep): A tf-agents object consisting of
                a step type, reward, discount factor, and observation.
            step (int): An integer denoting the current time step count.  Used \
                only to check if this is on the first time step of collecting
                an episode.  Defaults to 0.
            only_ego_car (bool): Whether to gather a time step for only the
                'ego car' (the car currently being trained/evaluated).  Defaults
                to False.
            ego_car_index (int): An integer between [0, NUM_AGENTS-1] indicating
                which observation and reward slices that will be taken from the
                overall time step.  Defaults to 0.
            max_steps (int):  The maximum number of steps an episode can be.
                Defaults to 1000.

        Returns:
            agent_ts (tf-agents TimeStep): Returns a tf-agents time step with
                the appropriate observation and reward slices.  This can be
                for all agents (if only_ego_car is set to False), or for all the
                cars (if only_ego_car is set to True).
        """
        # Extract discount and step type and convert to tf-tensors
        discount = time_step.discount
        if len(discount.numpy().shape) > 1 or discount.numpy().shape[0] > 1:
            discount = discount[0]
        discount = tf.convert_to_tensor(discount, dtype=tf.float32,
                                        name='discount')

        # Set step type (same for all agents)
        if step == 0:  # Start time step
            step_type = 0
        elif step == max_steps-1:  # Last time step
            step_type = 2
        else:  # Middle time step
            step_type = 1
        step_type = tf.convert_to_tensor([step_type], dtype=tf.int32,
                                         name='step_type')

        # Extract rewards for all agents
        try:
            R = [tf.convert_to_tensor(time_step.reward[:, car_id],
                                      dtype=tf.float32, name='reward') \
                 for car_id in range(self.num_agents)]
        except:
            R = [tf.convert_to_tensor(time_step.reward,
                                      dtype=tf.float32, name='reward') \
                 for _ in range(self.num_agents)]

        # Extract time step for ego agent only
        if only_ego_car:
            processed_observation = \
                self._process_observations(time_step, only_ego_car=only_ego_car,
                                           ego_car_index=ego_car_index)
            return ts.TimeStep(step_type, R[ego_car_index], discount,
                                   tf.convert_to_tensor(processed_observation,
                                   dtype=tf.float32, name='observations'))

        else:
            processed_observations = \
                self._process_observations(time_step, only_ego_car=only_ego_car,
                                           ego_car_index=ego_car_index)
            return [ts.TimeStep(step_type, R[car_id], discount,
                    tf.convert_to_tensor(processed_observations[car_id],
                                         dtype=tf.float32, name='observations'))
                    for car_id in range(self.num_agents)]

    def _process_observations(self, time_step, only_ego_car=False,
                              ego_car_index=0):
        """Helper function for processing observations for multi-agent and
        multi-frame configurations.

        Arguments:
            time_step (tf-agents TimeStep): A tf-agents object consisting of
                a step type, reward, discount factor, and observation.
            only_ego_car (bool): Whether to gather a time step for only the
                'ego car' (the car currently being trained/evaluated).  Defaults
                to False.
            ego_car_index (int): An integer between [0, NUM_AGENTS-1] indicating
                which observation and reward slices that will be taken from the
                overall time step.  Defaults to 0.

        Returns:
            processed_observation (tf.Tensor):  Tensor corresponding to the
                processed observation across all the observing agents.
        """
        # Extract time step for ego agent only
        if only_ego_car:
            input_observation = time_step.observation[:, ego_car_index]

            if self.num_agents > 1:  # Multi-car
                wrapper = self.observation_wrappers[ego_car_index]
            else:  # Single car
                wrapper = self.observation_wrapper

            # Process observations
            processed_observation = wrapper.get_obs_and_step(input_observation)

            return tf.convert_to_tensor(processed_observation, dtype=tf.float32,
                                        name='observations')

        # Extract a list of time steps for all cars
        else:
            input_observations = [time_step.observation[:, car_id] for
                                  car_id in range(self.num_agents)]
            if self.num_agents > 1:  # Multi-car
                processed_observations = \
                    [wrapper.get_obs_and_step(input_observation)
                     for wrapper, input_observation in
                     zip(self.observation_wrappers, input_observations)]

            else:  # Single car
                processed_observations = \
                    [self.observation_wrapper.get_obs_and_step(
                        input_observations[0])]

            return [tf.convert_to_tensor(processed_observations[car_id],
                                    dtype=tf.float32, name='observations')
                    for car_id in range(self.num_agents)]

    def collect_step(self, step=0, ego_car_index=0, use_greedy=False,
                     add_to_video=False):
        """Take a step in the training environment and add it to replay buffer
        for training PPO agent(s).

        Function for collecting a single time step from the environment.  Used
        for adding trajectories to the replay buffer that are used to train the
        agent(s).  Resets on the first time step - indicating the start of a
        new episode.

        Arguments:
            step (int): The current step of the episode.  Important for
                determining whether or not the environment needs to be reset
                and for tracking the training trajectories in tensorboard
                (if tensorboard plotting is enabled).  Defaults to 0.
            ego_car_index (int): An integer between [0, NUM_AGENTS-1] indicating
                which observation and reward slices that will be taken from the
                overall time step.  Defaults to 0.
            use_greedy (bool): Whether or not the agent's greedy policy is used.
                Defaults to False.
            add_to_video (bool): Whether to add the current time step
                observations to a video that is saved to the local filesystem
                (i.e. not tensorboard).  Defaults to False.

        Returns:
            reward (int): The reward for the 'ego car'.
        """
        # Get current time step
        time_step = self.train_env.current_time_step()

        # Create empty list of actions
        actions = []

        # Extract all individual agent time steps
        agent_timesteps = self.get_agent_timesteps(time_step, step=step,
                                                   only_ego_car=False,
                                                   max_steps=self.max_eval_episode_steps-1)

        # Determine which policy to use for ego agent
        if self.use_separate_agents:
            ego_agent_policy = self.collect_policies[ego_car_index]
        else:
            ego_agent_policy = self.collect_policy


        # Iterate through cars and select actions
        for car_id in range(NUM_AGENTS):

            # If the current policy is used to select action
            if car_id == ego_car_index:
                ego_agent_ts = agent_timesteps[car_id]
                ego_action_step = ego_agent_policy.action(ego_agent_ts)
                if use_greedy:
                    actions.append(ego_action_step.info['loc'])  # Greedy mean
                else:
                    actions.append(ego_action_step.action)  # Noisy mean

                # Add observation to video, if enabled
                if self.add_to_video:
                    rendered_state = time_step.observation[:, car_id].numpy()
                    if self.stacked_channels > 3:  # Frame stacking
                        rendered_state = rendered_state[:, :, :, :3]  # First frame
                    self.video_train.append(rendered_state)

            # Select greedy action for other cars from N-1 policies
            elif self.use_separate_agents:
                other_agent_ts = agent_timesteps[car_id]
                action_step = self.eval_policies[car_id].action(other_agent_ts)
                actions.append(action_step.action)

            # Select greedy action for other cars from one master policy
            elif self.use_self_play:
                other_agent_ts = agent_timesteps[car_id]
                action_step = self.eval_policy.action(other_agent_ts)
                actions.append(action_step.action)

        # Or add all observations using tensorboard
        if self.use_tensorboard:
            processed_observations = self._process_observations(time_step,
                                                               only_ego_car=False)
            self.tb_gif_train[step] = tf.convert_to_tensor(processed_observations)

        # Create the tf action tensor
        action_tensor = tf.convert_to_tensor([tf.stack(tuple(actions), axis=1)])

        # Step train env for next ts, and get next time step for ego agent agent
        next_time_step = self.train_env.step(action_tensor)
        ego_agent_next_ts = self.get_agent_timesteps(next_time_step, step=step+1,
                                                     ego_car_index=ego_car_index,
                                                     only_ego_car=True,
                                                     max_steps=self.collect_steps_per_episode-1)

        # Create trajectory for ego car and write it to replay buffer
        traj = trajectory.from_transition(ego_agent_ts, ego_action_step,
                                          ego_agent_next_ts)
        self.replay_buffer.add_batch(traj)

        # Add observation to video, if enabled
        if add_to_video:
            rendered_state = time_step.observation[:, ego_car_index].numpy()
            if self.num_frames > 1:  # Frame stacking
                rendered_state = rendered_state[:, :, :, :3]  # First frame
            self.video_train.append(rendered_state)

        return float(ego_agent_ts.reward)

    def collect_episode(self, epoch=0, ego_car_index=0, add_to_video=False):
        """ Collect a trajectory of experience for training the agent(s).

        Function for generating experience data for the replay buffer.
        Calls collect_step() above to add trajectories from the environment to
        the replay buffer in an episodic fashion.  Trajectories from the replay
        buffer are then used for training the agent using PPO actor-critic
        optimization approaches.

        Arguments:
             epoch (int): The current epoch of training.  Used for tracking
                the training trajectories in tensorboard (if tensorboard
                plotting is enabled).  Defaults to 0.
             ego_car_index (int): An integer between [0, NUM_AGENTS-1] indicating
                which observation and reward slices that will be taken from the
                overall time step.  Defaults to 0.
             add_to_video (bool): Whether to add the current time step
                observations to a video that is saved to the local filesystem
                (i.e. not tensorboard).  Defaults to False.
        """
        episode_reward = 0  # Placeholder reward
        step = 0  # Placeholder step count

        # Reset the environment
        self.train_env.reset()

        # Decide whether to use training or greedy policy for this episode
        use_greedy = float(np.random.binomial(n=1, p=self.epsilon))


        # Iteratively call collect_step to add trajectories to replay buffer
        while step < self.collect_steps_per_episode and \
                not self.is_last(mode='train'):
            episode_reward += self.collect_step(add_to_video=add_to_video,
                                                step=step, use_greedy=use_greedy,
                                                ego_car_index=ego_car_index)
            step += 1

        # Update global step count when training is finished
        self.global_step += step

        # Log average training return to tensorboard and create video gif
        if self.use_tensorboard:
            with self.tb_file_writer.as_default():
                tf.summary.scalar("Average Training Reward", float(episode_reward),
                                  step=self.global_step)
                frames = self.tb_gif_train
                video_summary("train/grid", frames, fps=self.FPS,
                              step=self.global_step, channels=self.num_channels)

            # Reset the tensorboard gif array
            self.tb_gif_train = np.zeros((self.collect_steps_per_episode,
                                         self.num_agents, self.H, self.W,
                                          self.stacked_channels))

    def compute_average_reward(self, ego_car_index=0):
        """ Compute average reward for agent(s) using greedy policy and
        evaluation environments.

        Function for computing the average reward over a series of evaluation
        episodes by creating simulation episodes using the agent's current
        policies, then computing rewards from taking actions using the
        evaluation (greedy) policy and averaging them.

        Arguments:
            ego_car_index (int): An integer between [0, NUM_AGENTS-1] indicating
                which observation and reward slices that will be taken from the
                overall time step.  Defaults to 0.

        Returns:
            episode_return (float): A float representing the average reward
                over the interval of episodes which the agent's policies are
                evaluated.
        """
        total_return = 0.0  # Placeholder reward

        # Evaluation loop
        for e in range(self.num_eval_episodes):
            time_step = self.eval_env.reset()

            # Initialize step counter and episode_return
            i = 0
            episode_return = 0.0
            while i < self.max_eval_episode_steps and \
                    not self.is_last(mode='eval'):
                # Create empty list of actions
                actions = []

                # Get all agent time steps
                agent_timesteps = self.get_agent_timesteps(time_step, step=i)

                # Iterate through cars and select actions
                for car_id in range(self.num_agents):

                    # If the current policy is used to select an action
                    if car_id == ego_car_index:  # Ego agent ts and video frame
                        ego_agent_ts = agent_timesteps[car_id]
                        rendered_state = ego_agent_ts.observation.numpy()
                        if self.num_frames > 1:  # Frame stacking
                            rendered_state = rendered_state[..., :3]  # First frame
                        self.video_eval.append(rendered_state)

                        if self.use_separate_agents:  # Use N different PPO agents
                            ego_action_step = self.eval_policies[car_id].action(ego_agent_ts)
                            actions.append(ego_action_step.action)   # Greedy policy

                        elif self.use_self_play:  # Use one master PPO agent
                            ego_action_step = self.eval_policy.action(ego_agent_ts)
                            actions.append(ego_action_step.action)  # Greedy policy

                    # Select greedy action for other cars from N-1 policies
                    elif self.use_separate_agents:
                        other_agent_ts = agent_timesteps[car_id]
                        action_step = self.eval_policies[car_id].action(other_agent_ts)
                        actions.append(action_step.action)

                    # Select greedy action for other cars from one master policy
                    elif self.use_self_play:
                        other_agent_ts = agent_timesteps[car_id]
                        action_step = self.eval_policy.action(other_agent_ts)
                        actions.append(action_step.action)

                # Create the tf action tensor
                action_tensor = tf.convert_to_tensor([tf.stack(tuple(actions),
                                                               axis=1)])

                # Step through with all actions
                time_step = self.eval_env.step(action_tensor)

                # Or add all observations using tensorboard
                if self.use_tensorboard:
                    processed_observations = self._process_observations(time_step,
                                                                        only_ego_car=False)
                    self.tb_gif_eval[i] = tf.convert_to_tensor(processed_observations)

                episode_return += ego_agent_ts.reward  # Add to reward
                if i % 250 == 0:
                    action = ego_action_step.action.numpy()
                    print("Action: {}, "
                          "Reward: {}".format(action, episode_return))
                i += 1

            print("Steps in episode: {}".format(i))
            total_return += episode_return
        avg_return = total_return / self.num_eval_episodes

        # If using tensorboard, create video
        if self.use_tensorboard:
            with self.tb_file_writer.as_default():
                video_summary("eval/grid".format(car_id), self.tb_gif_eval,
                              fps=self.FPS, step=self.global_step,
                              channels=self.num_channels)

            # Reset the tensorboard gif array
            self.tb_gif_eval = np.zeros((self.max_eval_episode_steps,
                                        self.num_agents, self.H, self.W,
                                         self.stacked_channels))

        print("Average return: {}".format(avg_return))

        # Append to aggregate returns

        if self.use_separate_agents:  # Computing the reward over multiple PPO agents
            self.eval_returns[ego_car_index].append(avg_return)
        else:  # Only one PPO agent to consider
            self.eval_returns.append(avg_return)

        return avg_return

    def collect_step_lstm(self, step=0, ego_car_index=0, add_to_video=False,
                          policy_states=None):
        """ Take a step in the training environment and add it to replay buffer
        for training PPO agent(s).  Uses LSTM-based actor and critic neural
        networks.

        Function for collecting a single time step from the environment.  Used
        for adding trajectories to the replay buffer that are used to train the
        agent(s).  Resets on the first time step - indicating the start of a
        new episode.

        Arguments:
            step (int): The current step of the episode.  Important for
                determining whether or not the environment needs to be reset
                and for tracking the training trajectories in tensorboard
                (if tensorboard plotting is enabled).  Defaults to 0.
            ego_car_index (int): An integer between [0, NUM_AGENTS-1] indicating
                which observation and reward slices that will be taken from the
                overall time step.  Defaults to 0.
            use_greedy (bool): Whether or not the agent's greedy policy is used.
                Defaults to False.
            add_to_video (bool): Whether to add the current time step
                observations to a video that is saved to the local filesystem
                (i.e. not tensorboard).  Defaults to False.
            policy_states (dict):  A dictionary mapping agent ID(s) to policy
                states, in this case, LSTM hidden states.

        Returns:
            reward (int): The reward for the 'ego car'.
        """
        # Get current time step
        time_step = self.train_env.current_time_step()

        # Create empty list of actions and next policy states
        actions = []
        next_policy_states = {}

        # Extract all individual agent time steps
        agent_timesteps = self.get_agent_timesteps(time_step, step=step,
                                                   only_ego_car=False)

        # Determine which policy to use for ego agent
        if self.use_separate_agents:
            ego_agent_policy = self.collect_policies[ego_car_index]
        else:
            ego_agent_policy = self.collect_policy

        # Iterate through cars and select actions
        for car_id in range(self.num_agents):
            # If the current policy is used to select action
            if car_id == ego_car_index:
                ego_agent_ts = agent_timesteps[car_id]
                if self.use_separate_agents:
                    ego_policy_step = ego_agent_policy.action(
                        ego_agent_ts, policy_states[car_id])
                else:
                    ego_policy_step = self.collect_policy.action(ego_agent_ts,
                                                                 policy_states[car_id])
                if use_greedy:
                    actions.append(ego_action_step.info['loc'])  # Greedy mean
                else:
                    actions.append(ego_action_step.action)  # Noisy mean
                policy_state = ego_policy_step.state

                # Add observation to video, if enabled
                if self.add_to_video:
                    rendered_state = time_step.observation[:, car_id].numpy()
                    if self.num_frames > 1:  # Frame stacking
                        rendered_state = rendered_state[..., :3]  # First frame
                    self.video_train.append(rendered_state)

            # Select greedy action for other cars from N-1 policies
            elif self.use_separate_agents:
                other_agent_ts = agent_timesteps[car_id]
                policy_step = self.eval_policies[car_id].action(other_agent_ts, policy_states[car_id])
                policy_state = policy_step.state
                actions.append(policy_step.action)  # Add action to agent tensor

            # Select greedy action for other cars from one master policy
            elif self.use_self_play:
                other_agent_ts = agent_timesteps[car_id]
                policy_step = self.eval_policy.action(other_agent_ts, policy_states[car_id])
                policy_state = policy_step.state
                actions.append(policy_step.action)

            # Add the new policy state for each agent
            next_policy_states[car_id] = policy_state  # Get next policy state

        # Or add all observations using tensorboard
        if self.use_tensorboard:
            processed_observations = self._process_observations(time_step,
                                                                only_ego_car=False)
            self.tb_gif_train[step] = tf.convert_to_tensor(processed_observations)

        # Create the tf action tensor
        action_tensor = tf.convert_to_tensor([tf.stack(tuple(actions), axis=1)])

        # Step train env for next ts, and get next ts for ego agent agent
        next_time_step = self.train_env.step(action_tensor)
        ego_agent_next_ts = self.get_agent_timesteps(next_time_step,
                                                     step=step + 1,
                                                     ego_car_index=ego_car_index,
                                                     only_ego_car=True)

        # Create trajectory for ego car and write it to replay buffer
        traj = trajectory.from_transition(ego_agent_ts, ego_policy_step,
                                          ego_agent_next_ts)
        self.replay_buffer.add_batch(traj)

        # Add observation to video, if enabled
        if add_to_video:
            rendered_state = time_step.observation[:, ego_car_index].numpy()
            if self.num_frames > 1:  # Frame stack
                rendered_state = rendered_state[:, :, :, 3]  # First frame
            self.video_train.append(rendered_state)

        return next_policy_states, float(ego_agent_ts.reward)

    def reset_policy_states(self, ego_car_index=0, mode='train'):
        """ Reset the policy state(s) of the PPO agent(s).

        Function for resetting the policy state(s) of the PPO agent(s) by
        resetting the hidden states of the LSTMs for the actor and critic
        neural networks.  Note that this function is only applicable for
        LSTM-based policies.

        Arguments:
            ego_car_index (int): An integer between [0, NUM_AGENTS-1] indicating
                which observation and reward slices that will be taken from the
                overall time step.  Defaults to 0.
            mode (str): Whether to check if the 'train' environment or 'eval'
                environment is currently on its last time step for an episode.
                Defaults to 'train' (the training environment).

        Returns:
            policy_states (dict):  A dictionary mapping agent ID(s) to policy
                states, in this case, LSTM hidden states.
        """
        if mode == 'train':
            if self.use_separate_agents:
                policy_states = {car_id: self.eval_policies[car_id].get_initial_state(
                    self.train_env.batch_size) for car_id in range(self.num_agents)}
                policy_states[ego_car_index] = self.collect_policies[
                    ego_car_index].get_initial_state(self.train_env.batch_size)
            else:
                policy_states = {car_id: self.eval_policy.get_initial_state(self.train_env.batch_size)
                                 for car_id in range(self.num_agents)}
                policy_states[ego_car_index] = self.collect_policy.get_initial_state(
                    self.train_env.batch_size)

        elif mode == 'eval':
            if self.use_separate_agents:
                policy_states = {
                    car_id: self.eval_policies[car_id].get_initial_state(
                        self.eval_env.batch_size) for car_id in
                    range(self.num_agents)}
            else:
                policy_states = {car_id: self.eval_policy.get_initial_state(
                    self.eval_env.batch_size) for car_id in
                    range(self.num_agents)}

        return policy_states

    def collect_episode_lstm(self, epoch=0, ego_car_index=0, add_to_video=False):
        """ Collect a trajectory of experience for training the agent(s).  Used
        for LSTM-based agents.

        Function for generating experience data for the replay buffer.
        Calls collect_step() above to add trajectories from the environment to
        the replay buffer in an episodic fashion.  Trajectories from the replay
        buffer are then used for training the agent using PPO actor-critic
        optimization approaches.  This function should only be used for
        LSTM-based policies.

        Arguments:
             epoch (int): The current epoch of training.  Used for tracking
                the training trajectories in tensorboard (if tensorboard
                plotting is enabled).  Defaults to 0.
             ego_car_index (int): An integer between [0, NUM_AGENTS-1] indicating
                which observation and reward slices that will be taken from the
                overall time step.  Defaults to 0.
             add_to_video (bool): Whether to add the current time step
                observations to a video that is saved to the local filesystem
                (i.e. not tensorboard).
        """
        # Get initial policy states for agents
        policy_states = self.reset_policy_states(ego_car_index=ego_car_index)

        episode_reward = 0  # Placeholder reward
        step = 0  # Placeholder time step count

        # Reset the environment
        self.train_env.reset()

        # Decide whether to use training or greedy policy for this episode
        use_greedy = float(np.random.binomial(n=1, p=self.epsilon))

        while step < self.collect_steps_per_episode and \
                not self.is_last(mode='train'):
            if step % 1000 == 0:
                print("Step number: {}".format(step))
            policy_states, ego_reward = self.collect_step_lstm(add_to_video=add_to_video,
                                                               step=step, use_greedy=use_greedy,
                                                               ego_car_index=ego_car_index,
                                                               policy_states=policy_states)
            # Update reward and step counter
            episode_reward += ego_reward
            step += 1

        # Update global step count when training is finished
        self.global_step += step

        # Log average training return to tensorboard and create eval gif
        if self.use_tensorboard:
            with self.tb_file_writer.as_default():
                tf.summary.scalar("Average Training Reward", float(episode_reward),
                                  step=self.global_step)
                frames = self.tb_gif_train
                video_summary("train/grid", frames, fps=self.FPS,
                              step=self.global_step,
                              channels=self.num_channels)

            # Reset the tensorboard gif array
            self.tb_gif_train = np.zeros((self.collect_steps_per_episode,
                                          self.num_agents, self.H, self.W,
                                          self.stacked_channels))

    def compute_average_reward_lstm(self, ego_car_index=0):
        """ Compute average reward for agent(s) using greedy policy and
        evaluation environments.

        Function for computing the average reward over a series of evaluation
        episodes by creating simulation episodes using the agent's current
        policies, then computing rewards from taking actions using the
        evaluation (greedy) policy and averaging them.

        Arguments:
            ego_car_index (int): An integer between [0, NUM_AGENTS-1] indicating
                which observation and reward slices that will be taken from the
                overall time step.  Defaults to 0.

        Returns:
            episode_return (float): A float representing the average reward
                over the interval of episodes which the agent's policies are
                evaluated.
        """
        total_return = 0.0

        # Evaluation loop
        for _ in range(self.num_eval_episodes):
            time_step = self.eval_env.reset()

            # Initialize step counter and episode_return
            i = 0
            episode_return = 0.0

            # Get initial policy states for agents
            policy_states = self.reset_policy_states(ego_car_index=ego_car_index,
                                                     mode='eval')

            while i < self.max_eval_episode_steps and \
                    not self.is_last(mode='eval'):

                # Create empty list of actions
                actions = []

                # Get all agent time steps
                agent_timesteps = self.get_agent_timesteps(time_step, step=i)

                # Iterate through cars and select actions
                for car_id in range(NUM_AGENTS):

                    # If a current policy is used to select an action
                    if car_id == ego_car_index:  # Ego agent ts and video frame
                        ego_agent_ts = agent_timesteps[car_id]
                        rendered_state = ego_agent_ts.observation.numpy()
                        if self.num_frames > 1:  # Frame stacking
                            rendered_state = rendered_state[..., :3]  # First frame
                        self.video_eval.append(rendered_state)

                        if self.use_separate_agents:  # Use N different PPO agents
                            ego_policy_step = self.eval_policies[car_id].action(ego_agent_ts,
                                                                                policy_states[car_id])
                            actions.append(ego_policy_step.action)   # Greedy policy

                        elif self.use_self_play:  # Use one master PPO agent
                            ego_policy_step = self.eval_policy.action(ego_agent_ts,
                                                                      policy_states[car_id])
                            actions.append(ego_policy_step.action)  # Greedy policy

                        policy_state = ego_policy_step.state

                    # Select greedy action for other cars from N-1 policies
                    elif self.use_separate_agents:
                        other_agent_ts = agent_timesteps[car_id]
                        policy_step = self.eval_policies[car_id].action(other_agent_ts,
                                                                        policy_states[car_id])
                        actions.append(policy_step.action)
                        policy_state = policy_step.state

                    # Select greedy action for other cars from one master policy
                    elif self.use_self_play:
                        other_agent_ts = agent_timesteps[car_id]
                        policy_step = self.eval_policy.action(other_agent_ts,
                                                              policy_states[car_id])
                        actions.append(policy_step.action)
                        policy_state = policy_step.state

                    # Update policy states for next step
                    policy_states[car_id] = policy_state

                # Create the tf action tensor
                action_tensor = tf.convert_to_tensor([tf.stack(tuple(actions), axis=1)])

                # Step through with all actions
                time_step = self.eval_env.step(action_tensor)

                # Or add all observations using tensorboard
                if self.use_tensorboard:
                    processed_observations = self._process_observations(time_step,
                                                                        only_ego_car=False)
                    self.tb_gif_eval[i] = tf.convert_to_tensor(processed_observations)

                episode_return += ego_agent_ts.reward  # Add to reward
                if i % 250 == 0:
                    action = ego_policy_step.action.numpy()
                    print("Action: {}, "
                          "Reward: {}".format(action, episode_return))
                    print("POLICY STATES: {}".format(
                        [np.sum(policy_states[i]) for i
                         in range(self.num_agents)]))
                i += 1
            print("Steps in episode: {}".format(i))
            total_return += episode_return
        avg_return = total_return / self.num_eval_episodes

        # If using tensorboard, create video
        if self.use_tensorboard:
            with self.tb_file_writer.as_default():
                video_summary("eval/grid".format(car_id), self.tb_gif_eval,
                              fps=self.FPS, step=self.global_step,
                              channels=self.num_channels)

            # Reset the tensorboard gif array
            self.tb_gif_eval = np.zeros((self.max_eval_episode_steps,
                                         self.num_agents, self.H, self.W,
                                         self.stacked_channels))

        print("Average return: {}".format(avg_return))

        # Append to aggregate returns
        if self.use_separate_agents:  # Computing the reward over multiple PPO agents
            self.eval_returns[ego_car_index].append(avg_return)
        else:  # Only one PPO agent to consider
            self.eval_returns.append(avg_return)

        return avg_return

    def train_agent(self):
        """ Train the PPO agent using PPO actor-critic optimization and
        trajectories gathered from the replay buffer.

        Function for training a PPO tf-agent using trajectories from the replay
        buffer.  Does initial evaluation of the agent prior to training, and
        then iterates over epochs of the following procedure:

            a. Collect an episode of data, and write the trajectories to the
               replay buffer.

            b. Train from the trajectories on the replay buffer.  Updates the
               weights of the actor and value networks.

            c. Empty the replay buffer.

            d. (If enabled) Save data to disk for tensorboard.

            e. Depending on epoch number and the evaluation and logging
               intervals, evaluate the agent or log information.

        Returns:
            agent (PPO agent): The PPO agent trained during the training
                process.  If using multiple agents, returns all trained agents.
        """
        eval_epochs = []

        # Optimize by wrapping some of the code in a graph using TF function.
        if self.use_separate_agents:
            for car_id in range(self.num_agents):
                self.agents[car_id].train = common.function(self.agents[car_id].train)
                self.agents[car_id].train_step_counter.assign(0)
        else:
            self.agent.train = common.function(self.agent.train)

        # Compute pre-training returns
        if self.use_lstm:
            avg_return = self.compute_average_reward_lstm(ego_car_index=0)

        else:
            avg_return = self.compute_average_reward(ego_car_index=0)

        # Log average training return to tensorboard
        if self.use_tensorboard:
            with self.tb_file_writer.as_default():
                tf.summary.scalar("Average Eval Reward", float(avg_return),
                                  step=self.global_step)

        print("DONE WITH PRELIMINARY EVALUATION...")

        # Append for output plot, create video, and empty eval video array
        eval_epochs.append(0)
        self.create_video(mode='eval', ext=0)
        self.video_eval = []  # Empty to create a new eval video
        returns = [avg_return]

        # Reset the environment time step and global and episode step counters
        time_step = self.train_env.reset()
        step = 0
        i = 0

        # Training loop
        for i in range(self.epochs):

            # Train for maximum number of steps
            if self.global_step >= self.total_steps:
                print("Reached the end of training with {} training steps".format(self.global_step))
                break

            # Set the agent index for the ego car for this epoch
            ego_car_index = i % self.num_agents
            print("Training epoch: {}".format(i))

            # Collect trajectories for current ego car (rotates each epoch)
            print("Collecting episode for car with ID {}".format(ego_car_index))

            # Reset the old training video
            self.video_train = []

            # Collect data for car index i % self.num_agents
            if self.use_lstm:
                self.collect_episode_lstm(epoch=i, ego_car_index=ego_car_index)
                print("LSTM")

            else:
                self.collect_episode(epoch=i, ego_car_index=ego_car_index)
                print("No LSTM")

            # Create a training episode video every 100 episodes
            if i % 100 == 0 and self.add_to_video:
                self.create_video(mode='train', ext=i)  # Create video of training
            print("Collected Episode")

            # Whether or not to do an optimization step with the GPU
            if self.use_gpu:
                device = '/gpu:0'
            else:
                device = '/cpu:0'

            # Do computation on selected device above
            with tf.device(device):

                # Gather trajectories from replay buffer
                trajectories = self.replay_buffer.gather_all()

                # Train on N different PPO agents
                if self.use_separate_agents:

                    # Take training step - with observations and rewards of ego agent
                    train_loss = self.agents[ego_car_index].train(experience=trajectories)

                    # Log loss to tensorboard
                    if self.use_tensorboard:
                        with self.tb_file_writer.as_default():
                            tf.summary.scalar("Training Loss Agent {}".format(ego_car_index),
                                              float(train_loss.loss),
                                              step=self.global_step // self.num_agents)

                    # Step the counter, and log/evaluate agent
                    step = self.agents[ego_car_index].train_step_counter.numpy()

                # Train on a single PPO agent
                else:
                    # Take training step - with observations and rewards of ego agent
                    train_loss = self.agent.train(experience=trajectories)

                    # Log loss to tensorboard
                    if self.use_tensorboard:
                        with self.tb_file_writer.as_default():
                            tf.summary.scalar("Training Loss",
                                              float(train_loss.loss), step=self.global_step)

            with tf.device('/cpu:0'):

                if self.global_step % self.log_interval == 0:
                    print('step = {0}: loss = {1}'.format(self.global_step,
                                                          train_loss.loss))

                if i % self.eval_interval == 0:

                    # Compute average return
                    if self.use_lstm:
                        avg_return = self.compute_average_reward_lstm(ego_car_index=ego_car_index)
                    else:
                        avg_return = self.compute_average_reward(ego_car_index=ego_car_index)

                    # Log average eval return to tensorboard and store it
                    if self.use_tensorboard:
                        with self.tb_file_writer.as_default():
                            tf.summary.scalar("Average Eval Reward",
                                              float(avg_return),
                                              step=self.global_step)
                    eval_epochs.append(i + 1)
                    print(
                        'epoch = {0}: Average Return = {1}'.format(step, avg_return))
                    returns.append(avg_return)
                    if self.add_to_video:
                        self.create_video(mode='eval', ext=i)
                    self.video_eval = []  # Empty to create a new eval video

                # Save checkpoints every save_interval epochs
                if i % self.save_interval == 0 and i != 0:
                    self.save_policies(epochs_done=i)
                    print("Epochs: {}".format(i))

                # Clear the replay buffer for the next episode
                self.replay_buffer.clear()

        # At the end of training, return the agent(s)
        if self.use_separate_agents:
            return self.agents
        else:
            return self.agent

    def create_video(self, mode='eval', ext=0, ego_car_index=0):
        """ Creates .avi videos of the agents' observations during training
        and evaluation.

        Function for creating .avi videos for viewing episodes from agent
        training and evaluation.  Saves frames stored in the video arrays to
        a sequential .avi video inside the logging directory.

        Arguments:
            mode (str): The mode 'train' or 'eval' in which the class creates
                a video.  Pulls from self.video_train or self.video_eval,
                respectively.

            ext (str): The extension, denoting the episode and, if there
                are multiple agents, the agent numbers.

            ego_car_index (int): The integer car id index corresponding to
                the ego agent.
        """
        # Select mode to create video in
        if mode == 'eval':  # Plot episode from evaluation
            video = self.video_eval
        elif mode == 'train':  # Plot episode from training
            video = self.video_train

        # Check if video is zero length, and get sizes
        if len(video) == 0:
            raise AssertionError("Video is empty.")
        print("Number of frames in video: {}".format(len(video)))
        obs_size = video[0].shape
        width = np.uint(obs_size[-3])
        height = np.uint(obs_size[-2])
        channels = np.uint(obs_size[-1])
        print("HEIGHT IS: {}, WIDTH IS: {}, CHANNELS IS: {}".format(width, height, channels))

        # Videowriter objects for OpenCV
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        out_file = os.path.join(self.log_dir,
                                "trajectories_{}_epoch_{}_agent_{}"
                                ".avi".format(mode, ext, ego_car_index))
        out = cv.VideoWriter(out_file, fourcc, self.FPS, (width, height))

        # Add frames to output video
        for i in range(len(video)):
            img_rgb = cv.cvtColor(np.uint8(255 * video[i][0]),
                                  cv.COLOR_BGR2RGB)  # Save as RGB image
            out.write(img_rgb)
        out.release()

    def plot_eval(self):
        """ Plots average evaluation returns for an agent over time.

        Creates a matplotlib plot for the average evaluation returns for each
        agent over time.  This file is saved to the 'policy_save_dir', as
        specified in the constructor.
        """
        if self.use_separate_agents:  # Make graphs for N separate agents
            for car_id in range(self.num_agents):
                # TODO(rms): How to plot for multiple agents?
                xs = [i * self.eval_interval for
                      i in range(len(self.eval_returns[car_id]))]
                plt.plot(xs, self.eval_returns[car_id])
                plt.xlabel("Training epochs")
                plt.ylabel("Average Return")
                plt.title("Average Returns as a Function "
                          "of Training (Agent {})".format(car_id))
                save_path = os.path.join(self.policy_save_dir,
                                         "eval_returns_agent_{}"
                                         ".png".format(car_id))
                plt.savefig(save_path)
                print("Created plot of returns for agent {}...".format(car_id))

        else:
            xs = [i * self.eval_interval for i in range(len(self.eval_returns))]
            plt.plot(xs, self.eval_returns)
            plt.xlabel("Training epochs")
            plt.ylabel("Average Return")
            plt.title("Average Returns as a Function of Training")
            save_path = os.path.join(self.policy_save_dir, "eval_returns.png")
            plt.savefig(save_path)
            print("CREATED PLOT OF RETURNS")


    def save_policies(self, epochs_done=0, is_final=False):
        """ Save PPO policy objects to the local filesystem.

        Using the PolicySaver(s) defined in the trainer constructor, this
        function saves the training and evaluation policies according to
        policy_save_dir as specified in the constructor.

        Arguments:
            epochs_done (int):  The number of epochs completed in the
                training process at the time this save function is called.
        """
        # If final, just use 'FINAL'
        if is_final:
            epochs_done = "FINAL"

        # Multiple PPO agents
        if self.use_separate_agents:

            # Iterate through training policies and save each of them
            for i, train_saver in enumerate(self.train_savers):
                if custom_path is None:
                    train_save_dir = os.path.join(self.policy_save_dir, "train",
                                                 "epochs_{}".format(epochs_done),
                                                 "agent_{}".format(i))
                else:
                    train_save_dir = os.path.join(self.policy_save_dir, "train",
                                                  "epochs_{}".format(
                                                      custom_path),
                                                  "agent_{}".format(i))
                if not os.path.exists(train_save_dir):
                    os.makedirs(train_save_dir, exist_ok=True)
                train_saver.save(train_save_dir)

            print("Training policies saved...")

            # Iterate through eval policies
            for i, eval_saver in enumerate(self.eval_savers):
                eval_save_dir = os.path.join(self.policy_save_dir, "eval",
                                             "epochs_{}".format(epochs_done),
                                             "agent_{}".format(i))
                if not os.path.exists(eval_save_dir):
                    os.makedirs(eval_save_dir, exist_ok=True)
                eval_saver.save(eval_save_dir)

            print("Eval policies saved...")

        # Master PPO agent
        else:
            # Save training policy
            train_save_dir = os.path.join(self.policy_save_dir, "train",
                                          "epochs_{}".format(epochs_done))
            if not os.path.exists(train_save_dir):
                os.makedirs(train_save_dir, exist_ok=True)
            self.train_saver.save(train_save_dir)

            print("Training policy saved...")

            # Save eval policy
            eval_save_dir = os.path.join(self.policy_save_dir, "eval",
                                         "epochs_{}".format(epochs_done))
            if not os.path.exists(eval_save_dir):
                os.makedirs(eval_save_dir, exist_ok=True)
            self.eval_saver.save(eval_save_dir)

            print("Eval policy saved...")

        # Save parameters in a file
        agent_params = {'normalize_obs': self.train_env.normalize,
                        'use_lstm': self.use_lstm,
                        'frame_stack': self.use_multiple_frames,
                        'num_frame_stack': self.env.num_frame_stack,
                        'obs_size': self.size}

        # Save as pkl parameter file
        params_path = os.path.join(self.policy_save_dir, "parameters.pkl")
        with open(params_path, "w") as pkl_file:
            pickle.dump(agent_params, pkl_file)
        pkl_file.close()

    def load_saved_policies(self, eval_model_path=None, train_model_path=None):
        """ Load from policies stored in the local filesystem.

        Loads saved policies for an agent, depending on what paths are
        specified in the function call.  This method sets the agent's
        policies to be the new policies that are loaded below.

        Arguments:
            eval_model_path (str): A file path (relative or absolute) to the
                directory containing the policy that will be loaded as the
                evaluation policy of the agent.
            train_model_path (str): A file path (relative or absolute) to the
                directory containing the policy that will be loaded as the
                training policy of the agent.
        """
        # Load evaluation and/or training policies from path
        if eval_model_path is not None:
            self.eval_policy = tf.saved_model.load(eval_model_path)
            print("Loading evaluation policy from: {}".format(eval_model_path))

        if train_model_path is not None:
            self.collect_policy = tf.saved_model.load(train_model_path)
            print("Loading training policy from: {}".format(train_model_path))


def parse_args():
    """Argument-parsing function for running this code."""
    # Create command line args
    parser = argparse.ArgumentParser()

    # Environment behavior
    parser.add_argument("-n", "--num_agents", default=2, type=int,
                        help="Number of cars in the environment.")
    parser.add_argument("-size", "--size", required=False,
                        default="96",
                        help="The width and height of the observation window.")
    parser.add_argument("-direction", "--direction", type=str, default='CCW',
                        help="Direction in which agents traverse the track.")
    parser.add_argument("-random_direction", "--use_random_direction",
                        required=False, action='store_true',
                        help="Whether agents are trained/evaluated on "
                             "both CW and CCW trajectories across the track.")
    parser.add_argument("-backwards_flag", "--backwards_flag", required=False,
                        action="store_true",
                        help="Whether to render a backwards flag indicator when "
                             "an agent drives on the track backwards.")
    parser.add_argument("-h_ratio", "--h_ratio", type=float, default=0.25,
                        help="Default height location fraction for where car"
                             "is located in observation upon rendering.")
    parser.add_argument("-ego_color", "--use_ego_color", required=False,
                        action="store_true", default="Whether to render each "
                                                     "ego car in the same color.")

    # MARL behavior
    parser.add_argument("-self_play", "--use_self_play",
                        required=False, action="store_true",
                        help="Flag for whether to use a single master PPO agent.")
    parser.add_argument("-n_agents", "--use_separate_agents",
                        required=False, action="store_true",
                        help="Flag for whether to use a N PPO agents.")

    # Learning behavior
    parser.add_argument("-epochs", "--total_epochs", default=1000, type=int,
                        help="Number of epochs to train agent over.")
    parser.add_argument("-steps", "--total_steps", type=int, default=10e6,
                        help="Total number of training steps to take.")
    parser.add_argument("-collect_episode_steps", "--collect_steps_per_episode",
                        default=1000, type=int,
                        help="Number of steps to take per collection episode.")
    parser.add_argument("-eval_episode_steps", "--eval_steps_per_episode",
                        default=1000, type=int,
                        help="Number of steps to take per evaluation episode.")
    parser.add_argument("-eval_interval", "--eval_interval", default=10,
                        type=int,
                        help="Evaluate every time epoch % eval_interval = 0.")
    parser.add_argument("-eval_episodes", "--num_eval_episodes", default=5,
                        type=int,
                        help="Evaluate over eval_episodes evaluation episodes.")
    parser.add_argument("-lr", "--learning_rate", default=5e-8, type=float,
                        help="Learning rate for PPO agent(s).")
    parser.add_argument("-lstm", "--use_lstm", required=False, action="store_true",
                        help="Flag for whether to use LSTMs on actor and critic"
                             "networks of the PPO agent.")
    parser.add_argument("-eps", "--epsilon", type=float, default=0.0,
                        help="Probability of training on the greedy policy for a"
                             "given episode")

    # Logging behavior
    parser.add_argument("-si", "--save_interval", default=10, type=int,
                        help="Save policies every time epoch % eval_interval = 0.")
    parser.add_argument("-li", "--log_interval", default=1, type=int,
                        help="Log results every time epoch % eval_interval = 0.")
    parser.add_argument("-tb", "--use_tensorboard", required=False,
                        action="store_true", help="Log with tensorboard as well.")
    parser.add_argument("-add_to_video", "--add_to_video", required=False,
                        action="store_true",
                        help="Whether to save trajectories as videos.")

    # Directories
    parser.add_argument("-exp_name", "--experiment_name", type=str,
                        default="experiment_{}", required=False,
                        help="Name of experiment (for logging).")

    # Parse arguments
    args = parser.parse_args()

    # Display CLI choices and return
    print("Your selected training parameters: \n {}".format(vars(args)))
    return args


def main():
    """ Main function for creating a PPO agent and training it on the
    multi-player OpenAI Gym Car Racing simulator.
    """
    if USE_XVFB:  # Headless render\
        wrapper = Xvfb()
        wrapper.start()

    # Read in arguments from command line.
    if USE_CLI:

        # Parse args
        args = parse_args()

        # Create gym training environment using wrapper
        gym_train_env = MultiCarRacing(num_agents=args.num_agents,
                                     direction=args.direction,
                                     use_random_direction=args.use_random_direction,
                                     backwards_flag=args.backwards_flag,
                                     h_ratio=args.h_ratio,
                                     use_ego_color=args.use_ego_color)

        # Create gym evaluation environment using wrapper
        gym_eval_env = MultiCarRacing(num_agents=args.num_agents,
                                      direction=args.direction,
                                      use_random_direction=args.use_random_direction,
                                      backwards_flag=args.backwards_flag,
                                      h_ratio=args.h_ratio,
                                      use_ego_color=args.use_ego_color)

    # Use global default parameters
    else:
        # Create gym training environment using wrapper
        gym_train_env = MultiCarRacing(num_agents=NUM_AGENTS, direction=DIRECTION,
                                       use_random_direction=USE_RANDOM_DIRECTION,
                                       backwards_flag=BACKWARDS_FLAG,
                                       h_ratio=H_RATIO, use_ego_color=USE_EGO_COLOR)

        # Create gym evaluation environment using wrapper
        gym_eval_env = MultiCarRacing(num_agents=NUM_AGENTS, direction=DIRECTION,
                                       use_random_direction=USE_RANDOM_DIRECTION,
                                       backwards_flag=BACKWARDS_FLAG,
                                       h_ratio=H_RATIO,
                                       use_ego_color=USE_EGO_COLOR)


    gym_train_env.observation_space.dtype = np.float32  # For Conv2D data input
    gym_eval_env.observation_space.dtype = np.float32  # For Conv2D data input

    # Now create Python environment from gym env
    py_train_env = gym_wrapper.GymWrapper(gym_train_env)  # Gym --> Py
    py_eval_env = gym_wrapper.GymWrapper(gym_eval_env)    # Gym --> Py

    # Create training and evaluation TensorFlow environments
    tf_train_env = tf_py_environment.TFPyEnvironment(py_train_env)  # Py --> Tf
    tf_eval_env = tf_py_environment.TFPyEnvironment(py_eval_env)    # Py --> Tf

    # Display environment specs
    print("Observation spec: {} \n".format(tf_train_env.observation_spec()))
    print("Action spec: {} \n".format(tf_train_env.action_spec()))
    print("Time step spec: {} \n".format(tf_train_env.time_step_spec()))

    # Read in arguments from command line.
    if USE_CLI:

        # Instantiate and initialize a single agent for car(s)
        if USE_SELF_PLAY or NUM_AGENTS < 2:
            agents = make_agent(tf_train_env, lr=self.learning_rate,
                                use_lstm=self.use_lstm, size=args.size,
                                num_frames=args.num_frames,
                                num_channels=args.num_channels)
            agents.initialize()

        # Otherwise, make N agents for N cars
        else:
            agents = []
            for i in range(NUM_AGENTS):
                agents.append(make_agent(tf_train_env, lr=args.learning_rate,
                                         use_lstm=args.use_lstm, size=args.size,
                                         num_frames=args.num_frames,
                                         num_channels=args.num_channels))
                agents[i].initialize()

        # Instantiate the trainer
        trainer = PPOTrainer(agents, tf_train_env, tf_eval_env, size=args.size,
                             normalize=args.normalize, num_frames=args.num_frames,
                             num_channels=args.num_channels,
                             use_tensorboard=args.use_tensorboard,
                             add_to_video=args.add_to_video,
                             use_separate_agents=args.use_separate_agents,
                             use_self_play=args.use_self_play,
                             num_agents=args.num_agents, use_lstm=args.use_lstm,
                             experiment_name=args.experiment_name,
                             collect_steps_per_episode=args.collect_steps_per_episode,
                             total_epochs=args.total_epochs,
                             total_steps=args.total_steps,
                             eval_steps_per_episode=args.eval_steps_per_episode,
                             num_eval_episodes=args.num_eval_episodes,
                             eval_interval=args.eval_interval,
                             epsilon=args.epsilon,
                             save_interval=args.save_interval,
                             log_interval=args.log_interval)

    # Use global default parameters
    else:

        # Instantiate and initialize a single agent for car(s)
        if USE_SELF_PLAY or NUM_AGENTS < 2:
            agents = make_agent(tf_train_env, lr=LR, use_lstm=USE_LSTM,
                                size=SIZE, num_frames=NUM_FRAMES,
                                num_channels=NUM_CHANNELS)
            agents.initialize()

        # Otherwise, make N agents for N cars
        else:
            agents = []
            for i in range(NUM_AGENTS):
                agents.append(make_agent(tf_train_env, lr=LR, use_lstm=USE_LSTM,
                                         size=SIZE, num_frames=NUM_FRAMES,
                                         num_channels=NUM_CHANNELS))
                agents[i].initialize()

        # Instantiate the trainer
        trainer = PPOTrainer(agents, tf_train_env, tf_eval_env, size=SIZE,
                             normalize=NORMALIZE, num_frames=NUM_FRAMES,
                             num_channels=NUM_CHANNELS,
                             use_tensorboard=USE_TENSORBOARD,
                             add_to_video=ADD_TO_VIDEO,
                             use_separate_agents=USE_SEPARATE_AGENTS,
                             use_self_play=USE_SELF_PLAY,
                             num_agents=NUM_AGENTS, use_lstm=USE_LSTM,
                             experiment_name=EXPERIMENT_NAME,
                             collect_steps_per_episode=COLLECT_STEPS_PER_EPISODE,
                             total_epochs=TOTAL_EPOCHS, total_steps=TOTAL_STEPS,
                             eval_steps_per_episode=EVAL_STEPS_PER_EPISODE,
                             num_eval_episodes=NUM_EVAL_EPISODES,
                             eval_interval=EVAL_INTERVAL, epsilon=EPSILON,
                             save_interval=SAVE_INTERVAL,
                             log_interval=LOG_INTERVAL)

    print("Initialized agent, beginning training...")

    # Train agent, and when finished, save model
    trained_agent = trainer.train_agent()

    # Plot the total returns for all agents
    trainer.plot_eval()

    print("Training finished; saving agent...")

    trainer.save_policies()
    print("Policies saved.")

    if USE_XVFB:  # Headless render
        wrapper.stop()

if __name__ == "__main__":
    main()
