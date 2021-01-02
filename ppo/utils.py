""" Utility functions for baseline tf-agents PPO agent creation, training, and
tensorboard visualzation.  Includes:

1. Function for writing and processing video summaries for visualizing agent
performance in TensorFlow.

2. Class for observation processing and frame stacking
for PPO agents and PPO Racing Players.  Primarily leveraged as an attribute for other
PPO classes.  Assumes that the observations seen by the agents are TensorFlow
tensor based arrays.

3. Function for creating actor and critic networks used for PPO agents.

4. Function for creating PPO agents using actor and critic networks.
"""

# Tensorboard visualization and image processing
import numpy as np
import tensorflow as tf
from skimage.transform import resize
from external.dreamer.tools import encode_gif

# Network and agent creation
from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tf_agents.networks.actor_distribution_rnn_network import ActorDistributionRnnNetwork
from tf_agents.networks.value_network import ValueNetwork
from tf_agents.networks.value_rnn_network import ValueRnnNetwork
from tf_agents.agents.ppo import ppo_agent, ppo_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories.time_step import TimeStep


def video_summary(name, video, fps=50, step=0, channels=3):
    """ Create a gif-based video for visualizing agent observations/trajectories
    on tensorboard.

    Function for creating a video summary for Tensorboard.  Input is a numpy
    array consisting of observation frames for the agents.  See the shapes below
    for more implementation details.  Adapted from
    https://github.com/mit-drl/deep_latent_games/blob/master/external/dreamer/tools.py.

    The shape of these observations is: (n_frames, n_agents, height, width, RGB).

    Arguments:
        name (str): The name that will be written as a tensorboard image.
        video (np.array): A NumPy array corresponding to the frames of the agents'
            trajectories.
        fps (int): An integer corresponding to the frames per second of the
            output of the tensorboard GIF.
        step (int): The tensorboard global step associated with the video stack
            of observations.
        channels (int): How many channels are used in these observations.  If
            using a single frame, channels = 3; else if using multiple frames (4),
            channels = 12.
    """
    name = name if isinstance(name, str) else name.decode('utf-8')
    if channels == 3:  # Multiple color frames are used
        video = video[..., :3]  # Take first frame
    elif channels == 1:  # Grayscale
        grayscale = video[..., :1]
        video = np.repeat(grayscale, 3, axis=-1)

    if np.issubdtype(video.dtype, np.floating):
        video = np.clip(255 * video, 0, 255).astype(np.uint8)

    # Shape: (n_frames, n_agents, height, width, RGB)
    F, N, H, W, C = video.shape

    try:
        frames = video.reshape((F, N*H, W, C))
        summary = tf.compat.v1.Summary()
        image = tf.compat.v1.Summary.Image(height=F * H, width=W * N, colorspace=C)
        image.encoded_image_string = encode_gif(frames, fps)
        summary.value.add(tag=name + '/gif', image=image)
        tf.summary.experimental.write_raw_pb(summary.SerializeToString(), step)

    except (IOError, OSError) as e:
        print('GIF summaries require ffmpeg in $PATH.', e)
        frames = video.reshape((F, N*H, W, C))
        tf.summary.image(name + '/grid', frames, step)


class ObservationWrapper:
    """ Class for stacking and processing frame observations.

    This object is passed as an attribute to other PPO classes and RL frameworks
    that rely on image processing and frame stacking for generating observations.

    Arguments:
        size (tuple):  The size of the observation frames for each agent.
            Defaults to (96, 96).
        normalize (bool): Whether to map observations from [0,255] --> [0,1].
        num_channels (int):  Number of channels to use per frame.  Defaults to 3.
        num_frames (int):  The number of frames an agent is given for an
            observation.  If > 1, frame stacking is used, and the agent observes
            both the current frame and previous frame(s).  Defaults to 1.
    """
    def __init__(self, size=(96, 96), normalize=False, num_channels=3,
                 num_frames=1, num_agents=2):

        self.size = size  # Dimensions of observation frame
        self.normalize = normalize  # Normalize data from [0, 255] --> [0, 1]
        self.num_channels = num_channels  # 3 for RGB, 1 for greyscale
        self.num_frames = num_frames  # Number of frames in obs
        if self.num_frames > 1:  # Frame stacking
            self.frames = [tf.zeros(self.size + (self.num_channels,)) for i in
                                    range(self.num_frames)]  # Used as queue


    def get_obs_and_step(self, frame):
        """ Processes the observations from the environment.

        Processes the image-based observation data, and performs resizing/
        concatenation of observations according to the observation parameters
        provided in the constructor.

        Arguments:
            images (list):  Arrays corresponding to observations of the agent(s)
                in the environment, PRIOR to processing.

        Returns:
            images_pro (np.array): Arrays corresponding to observations of the
                agent(s) in the environment, AFTER processing.
        """
        processed_frame = self._process_image(tf.squeeze(frame))  # Process frame

        if self.num_frames == 1:  # Single-frame observations
            return processed_frame

        else:  # Frame stacking
            concat = [processed_frame] + self.frames[:-1]  # New frames list
            self.frames = concat  # Update frames
            stacked_frames = tf.concat(tuple(concat), axis=-1)  # Concatenate
            return stacked_frames

    def _process_image(self, image):
        """ Process each individual observation image.

        Processes the image-based observation data, and performs resizing/
        concatenation of observations according to the observation parameters
        provided in the constructor.  Processes each image, represented as a
        np.array, individually.

        Arguments:
            image (np.array):  Array corresponding to observations of a given
                agent in the environment, PRIOR to processing.

        Returns:
            image (np.array): Array corresponding to observations of a given
                agent in the environment, AFTER processing.
        """

        if self.num_channels == 1:  # grayscale
            image = tf.image.rgb_to_grayscale(image)

        elif self.num_channels == 3:  # rgb
            if len(tf.shape(tf.squeeze(image)).numpy()) < 3:  # If grayscale
                image = tf.repeat(tf.expand_dims(image, axis=-1),
                                  self.num_channels, axis=-1)  # gray --> rgb

        input_size = tuple(tf.shape(image)[:2].numpy())  # Image (width, height)
        if input_size != self.size:
            kwargs = dict(
                output_shape=self.size, mode='edge', order=1,
                preserve_range=True)

            # Resize the image according to the size parameter
            image = tf.convert_to_tensor(resize(image, **kwargs).astype(np.float32))
            if self.normalize and np.max(image) > 1.0:  # [0, 255] --> [0, 1]
                image = tf.divide(image, 255.0)
        return image

    def reset(self):
        """ Method for resetting the observed frames. """
        if self.num_frames > 1:
            self.frames = [tf.zeros(self.size + (self.num_channels,)) for i in
                                    range(self.num_frames)]  # Used as queue

def make_networks(env, size=(96, 96), num_frames=1, num_channels=3,
                  conv_params=[(16, 8, 4), (32, 3, 2)],
                  in_fc_params=(256,), out_fc_params=(128,), use_lstm=False,
                  lstm_size=(256,)):
    """ Creates the actor and critic neural networks of the PPO agent.

    Function for creating the neural networks for the PPO agent, namely the
    actor and value networks.

    Source for network params:
    https://www.arconsis.com/unternehmen/blog/reinforcement-learning-doom-with-tf-agents-and-ppo

    Arguments:
        env (TfPyEnvironment): A TensorFlow environment the agent interacts with.
        size (tuple):  The desired width and height of the observation space.
            Defaults to (96, 96).  Input tuple should preserve the original
            observation aspect ratio.
        num_frames (int):  Number of frames used in the agent's observation.
            Defaults to 1, num_frames > 1 indicates frame stacking.
        num_channels (int):  Number of color channels to include for each frame.
            Defaults to 3 (RGB), and 1 denotes grayscale.
        conv_params (list): A list corresponding to convolutional layer
            parameters for the PPO agent's actor and critic neural networks.
        in_fc_params (tuple): The number of neurons in the input fully
            connected layer of the actor and critic networks of the agent.
        out_fc_params (tuple): The number of neurons in the output fully
            connected layer of the actor and critic networks of the agent.
        use_lstm (bool):  Whether to use LSTM-based actor and critic networks.
        lstm_size (tuple): The number of hidden states inside the LSTM for the
            actor and critic networks of the agents.

    Returns:
        actor_net (ActorDistributionNetwork): A tf-agents Actor Distribution
            Network used for PPO agent action selection.
        value_net (ValueNetwork): A tf-agents Value Network used for
            PPO agent value estimation.
    """
    # Restructure time step spec to match expected processed observations
    processed_shape = tuple(size + (num_channels *num_frames,))
    obs_spec = env.observation_spec()  # Get old observation spec
    obs_spec = tensor_spec.BoundedTensorSpec(processed_shape, obs_spec.dtype,
                                             minimum=obs_spec.minimum,
                                             maximum=obs_spec.maximum,
                                             name=obs_spec.name)
    if use_lstm:  # LSTM-based policies
        # Define actor network
        actor_net = ActorDistributionRnnNetwork(obs_spec, env.action_spec(),
                                                conv_layer_params=conv_params,
                                                input_fc_layer_params=in_fc_params,
                                                lstm_size=lstm_size,
                                                output_fc_layer_params=out_fc_params)
        # Define value network
        value_net = ValueRnnNetwork(obs_spec, conv_layer_params=conv_params,
                                    input_fc_layer_params=in_fc_params,
                                    lstm_size=lstm_size,
                                    output_fc_layer_params=out_fc_params)

        print("Created Actor and Value Networks with LSTM...")

    else:  # non-LSTM-based policies
        # Define actor network
        actor_net = ActorDistributionNetwork(obs_spec, env.action_spec(),
                                             conv_layer_params=conv_params)
        # Define value network
        value_net = ValueNetwork(obs_spec, conv_layer_params=conv_params)

    return actor_net, value_net


def make_agent(env, lr=8e-5, use_lstm=False, size=(96, 96),
               num_frames=1, num_channels=3):
    """ Creates a tf-agents PPO agent.

    Function for creating the TensorFlow PPO agent using actor distribution
    and value networks according to the make_networks method above.  Can be
    used with or without LSTM networks.  The network archiecture used is
    actor-critic by nature.

    By default, this PPO agent uses an adaptive KL penalty divergence without
    clipping the importance ratio-weighted advantage, as well as an L2 loss
    of the value estimates of the network with a weighted coefficient of 0.5.

    Arguments:
        env (TfPyEnvironment): A TensorFlow environment the agent interacts with
            via observations and action selection.
        lr (float): The learning rate for the PPO agent.  Defaults to 8e-5.
        use_lstm (bool):  Whether an LSTM-based neural network architecture
            is used.  Defaults to False.
        size (tuple):  The desired width and height of the observation space.
            Defaults to (96, 96).  Input tuple should preserve the original
            observation aspect ratio.
        num_frames (int):  Number of frames used in the agent's observation.
            Defaults to 1, num_frames > 1 indicates frame stacking.
        num_channels (int):  Number of color channels to include for each frame.
            Defaults to 3 (RGB), and 1 denotes grayscale.

    Returns:
        agent (PPOAgent): A PPO agent used for learning behaviors
            in the environment given by env.
    """
    # Create the actor-critic networks, either with or without a LSTM network
    actor_net, value_net = make_networks(env, size=size, num_frames=num_frames,
                                         num_channels=num_channels,
                                         use_lstm=use_lstm)

    # Now create the PPO agent using the actor and value networks
    ts_spec = env.time_step_spec()
    processed_shape = tuple(size + (num_frames * num_channels,))

    # Create processed observation spec
    obs_spec = ts_spec.observation  # Get old observation spec
    obs_spec = tensor_spec.BoundedTensorSpec(processed_shape, obs_spec.dtype,
                                             minimum=obs_spec.minimum,
                                             maximum=obs_spec.maximum,
                                             name=obs_spec.name)

    # Create processed time step spec
    time_step_spec = TimeStep(discount=ts_spec.discount,
                              observation=obs_spec,
                              reward=ts_spec.reward,
                              step_type=ts_spec.step_type)

    print("TIMESTEP SPEC: {}, TS SPEC: {}".format(time_step_spec.reward,
                                                  ts_spec.reward))

    print("CREATING AGENT...")
    print("--> OBSERVATION SPEC: {}".format(obs_spec.shape))
    print("--> ACTION SPEC: {}".format(env.action_spec().shape))

    agent = ppo_agent.PPOAgent(time_step_spec, env.action_spec(),
                               actor_net=actor_net, value_net=value_net,
                               optimizer=tf.compat.v1.train.AdamOptimizer(lr))
    return agent