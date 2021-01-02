""" Loads saved PPO policies that can be used for evaluation.  Separate
evaluation from the main Elo Rating framework used to evaluate Dreamer and PPO
agents. """

# Native Python imports
import os
from datetime import datetime

# TensorFlow and tf-agents
from tf_agents.environments import gym_wrapper, tf_py_environment
import tf_agents.trajectories.time_step as ts

# Other external packages
import numpy as np
import tensorflow as tf

# Custom packages
from utils import video_summary, encode_gif
from openai_wrapper import OpenAIGym


def load_saved_policies(eval_model_path=None, train_model_path=None):
    """ Loads saved PPO policies from the local file system.

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

    Returns:
        collect_policy (TF Agents Policy):  A tf-agents policy object
            corresponding to the training/exploratory policy of the PPO agent.
        eval_policy (TF Agents Policy):  A tf-agents policy object
            corresponding to the evaluation/greedy policy of the PPO agent.
    """
    # Load evaluation and/or training policies from path
    if eval_model_path is not None:
        eval_policy = tf.saved_model.load(eval_model_path)
        print("Loading evaluation policy from: {}".format(eval_model_path))

    if train_model_path is not None:
        collect_policy = tf.saved_model.load(train_model_path)
        print("Loading training policy from: {}".format(train_model_path))

    return collect_policy, eval_policy


def get_agent_timesteps(time_step, only_ego_car=False,
                        ego_car_index=0, num_cars=2):
    """ Helper function for computing the time step for a specific agent.

    This function takes the time steps from the environment, and creates
    car-specific timesteps for use in tf-agents action selection.  This can
    be done for all cars in the environment, or for a single "ego car",
    depending on the arguments given.

    Arguments:
        time_step (tf-agents TimeStep): A tf-agents object consisting of
            a step type, reward, discount factor, and observation.

        only_ego_car (bool): A boolean flag if we want to take the ego car's
            observations.

        ego_car_index (int): An integer denoting our current time step.
            Used only to check if this is on the first time step of collecting
            an episode.  Defaults to 0.

        num_cars (int): The number of cars in the environment.  Defaults to 2.

    Returns:
        agent_ts (tf-agents TimeStep): Returns a tf-agents time step with
            the appropriate observation and reward slices (i.e. not the
            observations and rewards from all the agents).
    """
    # Extract discount and step type and convert to tf-tensors
    discount = time_step.discount
    if len(discount.numpy().shape) > 1 or discount.numpy().shape[0] > 1:
        discount = discount[0]

    discount = tf.convert_to_tensor(discount, dtype=tf.float32,
                                    name='discount')
    step_type = tf.convert_to_tensor([0], dtype=tf.int32,
                                     name='step_type')

    # Extract rewards for all agents
    try:
        R = [tf.convert_to_tensor(time_step.reward[:, car_id],
                                  dtype=tf.float32, name='reward')
                                  for car_id in range(num_cars)]
    except:
        R = [tf.convert_to_tensor(time_step.reward,
                                  dtype=tf.float32, name='reward')
                                  for _ in range(num_cars)]
    # Extract time step for ego agent only
    if only_ego_car:
        return ts.TimeStep(step_type, R[ego_car_index], discount,
                           tf.convert_to_tensor(
                               time_step.observation[:, ego_car_index],
                               dtype=tf.float32, name='observations'))

    # Extract a list of timesteps for all cars
    else:
        return [ts.TimeStep(step_type, R[car_id], discount,
                            tf.convert_to_tensor(
                                time_step.observation[:, car_id],
                                dtype=tf.float32, name='observations'))
                for car_id in range(num_cars)]


def is_last(env):
    """ Determines if the environment's time step is the 'last' time step.

    Function for determining if the time step is 'last' - i.e. at the end
    of an episode.  This is accomplished through the tf-agents API.

    Arguments:
        env (TfPyEnvironment):  Environment that the policies interact within.

    Returns:
        is_last (bool): Whether the current time step in the environment is the
            last time step.
    """
    step_types = env.current_time_step().step_type.numpy()

    # See if any time steps are 'last'
    is_last = bool(min(np.count_nonzero(step_types == 2), 1))

    return is_last


def compute_average_reward(policies, tb_file_writer, eval_env, eval_steps=1000,
                       ego_car_index=0, num_eval_episodes=5, num_cars=2,
                       self_play=True, H=64, W=64, FPS=50):
    """ Computes the average reward for trained policies in this environment.

    Function for computing the average reward over a series of evaluation
    episodes by creating simulation episodes using the agent's current
    policies, then computing rewards from taking actions using the
    evaluation (greedy) policy and averaging them.

    Arguments:
        policies (PPO Policies):  Policies that map states to deterministic
            action distributions.
            Loaded from saved locations in the computer file system.
        tb_file_writer (TensorBoard file writer):  Tensorboard file writer
            for visualizing episodes and summary statistics.
        eval_env (TfPyEnvironment):  Evaluation env used for loaded policies.
        eval_steps (int):  The number of evaluation steps taken in each
            episode of evaluation.  Defaults to 1000.
        ego_car_index (int): An integer between [0, NUM_AGENTS-1] indicating
            which observation and reward slices that will be taken from the
            overall time step.  Defaults to 0.
        num_eval_episodes (int):  The number of episodes over which we compute
            reward and evaluate our policies.
        num_cars (int):  The number of cars in the environment.
        self_play (bool):  Whether to use self-play (i.e. one agent makes
            all decisions).  If False, expects multiple policies.
        H, W (int, int):  The observation size of the environment.
        FPS (int):  The frames per second for generating Tensorboard GIFs.

    Returns:
        episode_return (float): A float representing the average reward
            over the episodes in which the agent's policies are evaluated.
    """
    total_return = 0.0  # Placeholder return value

    # Evaluation loop
    for e in range(num_eval_episodes):
        # Reset eval environment
        time_step = eval_env.reset()

        # Initialize step counter and episode_return
        i = 0
        episode_return = 0.0

        # Reset the tensorboard gif array
        tb_gif_eval = np.zeros((eval_steps, num_cars, H, W, 3))

        while not is_last(eval_env) and i < eval_steps:

            # Create empty list of actions
            actions = []

            # Get all agent timesteps
            agent_timesteps = get_agent_timesteps(time_step, step=i, num_cars=2)

            # Iterate through cars and select actions
            for car_id in range(num_cars):

                # If we use policy to select action
                if car_id == ego_car_index:  # Ego agent ts and video frame
                    ego_agent_ts = agent_timesteps[car_id]

                    if self_play:  # Use one master PPO agent
                        ego_action_step = policies[0].action(ego_agent_ts)
                        if i % 100 == 0:
                            print("ACTION: {}".format(ego_action_step.action))
                        actions.append(ego_action_step.action)  # Greedy policy

                    else:  # Use N different PPO agents
                        ego_action_step = policies[car_id].action(ego_agent_ts)
                        actions.append(ego_action_step.action)   # Greedy policy

                else:
                    # Select greedy action for other cars from N-1 policies
                    if self_play:
                        other_agent_ts = agent_timesteps[car_id]
                        action_step = policies[0].action(other_agent_ts)
                        actions.append(action_step.action)

                    # Select greedy action for other cars from one master policy
                    else:
                        other_agent_ts = agent_timesteps[car_id]
                        action_step = policies[0].action(other_agent_ts)
                        actions.append(action_step.action)

            # Create the tf action tensor
            action_tensor = tf.stack(tuple(actions), axis=1)

            # Step through with all actions
            time_step = eval_env.step(action_tensor)

            # Or add all observations using tensorboard
            tb_gif_eval[i] = time_step.observation

            episode_return += ego_agent_ts.reward  # Add to reward
            if i % 250 == 0:
                action = ego_action_step.action.numpy()
                print("Action: {}, "
                      "Reward: {}".format(action, episode_return))

            i += 1
        print("Steps in episode: {}".format(i))
        total_return += episode_return
    avg_return = total_return / num_eval_episodes

    # If using tensorboard, create video
    with tb_file_writer.as_default():
        video_summary("eval/grid".format(car_id), tb_gif_eval,
                      fps=FPS, step=e*eval_steps)

    print("Average return: {}".format(avg_return))

    return avg_return


def compute_average_reward_lstm(policies, tb_file_writer, eval_env, eval_steps=1000,
                            ego_car_index=0, num_eval_episodes=5, num_cars=2,
                            self_play=True, H=64, W=64, FPS=50):
    """ Computes the average reward for trained policies in this environment.

    Function for computing the average reward over a series of evaluation
    episodes by creating simulation episodes using the agent's current
    policies, then computing rewards from taking actions using the
    evaluation (greedy) policy and averaging them.

    Arguments:
        policies (PPO Policies):  Policies that map states and hidden states
            to deterministic action distributions.
            Loaded from saved locations in the computer file system.
        tb_file_writer (TensorBoard file writer):  Tensorboard file writer
            for visualizing episodes and summary statistics.
        eval_env (TfPyEnvironment):  Evaluation env used for loaded policies.
        eval_steps (int):  The number of evaluation steps taken in each
            episode of evaluation.  Defaults to 1000.
        ego_car_index (int): An integer between [0, NUM_AGENTS-1] indicating
            which observation and reward slices that will be taken from the
            overall time step.  Defaults to 0.
        num_eval_episodes (int):  The number of episodes over which we compute
            reward and evaluate our policies.
        num_cars (int):  The number of cars in the environment.
        self_play (bool):  Whether to use self-play (i.e. one agent makes
            all decisions).  If False, expects multiple policies.
        H, W (int, int):  The observation size of the environment.
        FPS (int):  The frames per second for generating Tensorboard GIFs.

    Returns:
        episode_return (float): A float representing the average reward
            over the episodes in which the agent's policies are evaluated.
    """
    total_return = 0.0

    # Evaluation loop
    for e in range(num_eval_episodes):
        time_step = eval_env.reset()

        # Initialize step counter and episode_return
        i = 0
        episode_return = 0.0

        # Reset the tensorboard gif array
        tb_gif_eval = np.zeros((eval_steps, num_cars, H, W, 3))

        # Get initial policy states for agents
        if not self_play:
            policy_states = {
                car_id: policies[car_id].get_initial_state(
                    eval_env.batch_size) for car_id in range(num_cars)}
        else:
            policy_states = {car_id: policies[0].get_initial_state(
                eval_env.batch_size) for car_id in range(num_cars)}

        while not is_last(eval_env) and i < eval_steps:

            # Create empty list of actions
            actions = []

            # Get all agent timesteps
            agent_timesteps = get_agent_timesteps(time_step, step=i)

            # Iterate through cars and select actions
            for car_id in range(num_cars):

                # If we use policy to select action
                if car_id == ego_car_index:  # Ego agent ts and video frame
                    ego_agent_ts = agent_timesteps[car_id]

                    if not self_play:  # Use N different PPO agents
                        ego_policy_step = policies[car_id].action(ego_agent_ts,
                                                                  policy_states[car_id])
                        actions.append(ego_policy_step.action)   # Greedy policy

                    else:  # Use one master PPO agent
                        ego_policy_step = policies[0].action(ego_agent_ts,
                                                                  policy_states[car_id])
                        actions.append(ego_policy_step.action)  # Greedy policy

                    policy_state = ego_policy_step.state

                # Select greedy action for other cars from N-1 policies
                elif not self_play:
                    other_agent_ts = agent_timesteps[car_id]
                    policy_step = policies[car_id].action(other_agent_ts,
                                                          policy_states[car_id])
                    actions.append(policy_step.action)
                    policy_state = policy_step.state

                # Select greedy action for other cars from one master policy
                else:
                    other_agent_ts = agent_timesteps[car_id]
                    policy_step = policies[0].action(other_agent_ts,
                                                          policy_states[car_id])
                    actions.append(policy_step.action)
                    policy_state = policy_step.state

                # Update policy states for next step
                policy_states[car_id] = policy_state

            # Create the tf action tensor
            action_tensor = tf.stack(tuple(actions), axis=1)

            # Step through with all actions
            time_step = eval_env.step(action_tensor)

            # Or add all observations using tensorboard
            tb_gif_eval[i] = time_step.observation

            episode_return += ego_agent_ts.reward  # Add to reward
            if i % 250 == 0:
                action = ego_policy_step.action.numpy()
                print("Action: {}, "
                      "Reward: {}".format(action, episode_return))
            i += 1
        print("Steps in episode: {}".format(i))
        total_return += episode_return

        # If using tensorboard, create video
        with tb_file_writer.as_default():
            video_summary("eval/grid".format(car_id), tb_gif_eval,
                          fps=FPS, step=eval_steps * e)

        # Reset the tensorboard gif array
        tb_gif_eval = np.zeros((eval_steps, num_cars, H, W, 3))
    avg_return = total_return / num_eval_episodes

    print("Average return: {}".format(avg_return))


def make_env():
    """ Function for creating the TfPyEnvironment from OpenAI Gym environment.
    """
    # Create wrapped environment
    gym_env = OpenAIGym("MultiCarRacing", size=(64, 64), normalize=True)
    gym_env.observation_space.dtype = np.float32  # For Conv2D data input

    # Now create Python and TensorFlow environments from gym env
    eval_env = tf_py_environment.TFPyEnvironment(
        gym_wrapper.GymWrapper(gym_env))

    # Display environment specs
    print("Observation spec: {} \n".format(train_env.observation_spec()))
    print("Action spec: {} \n".format(train_env.action_spec()))
    print("Time step spec: {} \n".format(train_env.time_step_spec()))
    print("Training env spec: {}".format(train_env.observation_spec()))

    return eval_env


def main():
    """ Main function for running evaluation.
    """
    USE_LSTM = True
    path_to_policies = os.path.join("models", "google", "models")
    EVAL_MODEL_PATH = os.path.join(path_to_policies, "self_play_20200506-150224",
                                   "eval", "epochs_5000")
    TRAIN_MODEL_PATH = os.path.join(path_to_policies,
                                    "self_play_20200506-150224",
                                    "train", "epochs_5000")
    EXPERIMENT_NAME = "self_play_video_recording_{}".format(datetime.now().strftime("%Y%m%d-%H%M%S"))
    LOG_DIR = os.path.join("exp_eval", EXPERIMENT_NAME)
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR, exist_ok=True)
    eval_env = make_env()
    train_policy, eval_policy = load_saved_policies(train_model_path=TRAIN_MODEL_PATH,
                                                    eval_model_path=EVAL_MODEL_PATH)
    print(eval_env.action_spec())
    tb_file_writer = tf.summary.create_file_writer(LOG_DIR)
    if USE_LSTM:
        compute_average_reward_lstm([train_policy], tb_file_writer, eval_env,
                                eval_steps=1000, ego_car_index=0,
                                num_eval_episodes=5, num_cars=2,
                                self_play=True, H=64, W=64, FPS=50)
    else:
        compute_average_reward([train_policy], tb_file_writer, eval_env,
                            eval_steps=1000, ego_car_index=0,
                            num_eval_episodes=5, num_cars=2,
                            self_play=True, H=64, W=64, FPS=50)

if __name__ == "__main__":
    main()
