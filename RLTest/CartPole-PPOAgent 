from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import numpy as np
import reverb

import tensorflow as tf


from tf_agents.metrics import tf_metrics
from tf_agents.agents.ppo import ppo_kl_penalty_agent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import value_network
from tf_agents.policies import py_tf_eager_policy
from tf_agents.replay_buffers import episodic_replay_buffer
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
env_name = "CartPole-v0"  # @param {type:"string"}
num_iterations = 2500  # @param {type:"integer"}
collect_episodes_per_iteration = 2  # @param {type:"integer"}
replay_buffer_capacity = 20000  # @param {type:"integer"}

fc_layer_params = (512, 256, 128, 64)

learning_rate = 1e-3  # @param {type:"number"}
log_interval = 250  # @param {type:"integer"}
num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 500  # @param {type:"integer"}


train_py_env = suite_gym.load(env_name,max_episode_steps=200)
eval_py_env = suite_gym.load(env_name,max_episode_steps=200)

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

actor_net = actor_distribution_network.ActorDistributionNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=fc_layer_params,
    kernel_initializer=tf.keras.initializers.VarianceScaling(
        scale=2.0, mode='fan_in', distribution='truncated_normal')
)

value_net = value_network.ValueNetwork(
    train_env.observation_spec()
)


optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)

tf_agent = ppo_kl_penalty_agent.PPOKLPenaltyAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    actor_net=actor_net,
    value_net=value_net,
    optimizer=optimizer,
    train_step_counter=train_step_counter,
    num_epochs=25,
    adaptive_kl_target=0.95,
    adaptive_kl_tolerance=0.1,
    initial_adaptive_kl_beta=1.0,
    kl_cutoff_factor=2.0,
    kl_cutoff_coef=1000.0)
tf_agent.initialize()

eval_policy = tf_agent.policy
collect_policy = tf_agent.collect_policy


def compute_avg_return(environment, policy, num_episodes=10):

    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


replay_buffer = episodic_replay_buffer.EpisodicReplayBuffer(
    tf_agent.collect_data_spec, capacity=replay_buffer_capacity, completed_only=True
)
stateful_replay_buffer = episodic_replay_buffer.StatefulEpisodicReplayBuffer(
    replay_buffer, num_episodes=1
)

replay_observer = [stateful_replay_buffer.add_batch]

# Dataset generates trajectories with shape [Bx2x...]
dataset = replay_buffer.as_dataset(
    sample_batch_size=64, num_steps=5, num_parallel_calls=3).prefetch(3)

# (Optional) Optimize by wrapping some of the code in a graph using TF function.
tf_agent.train = common.function(tf_agent.train)

# Reset the train step
tf_agent.train_step_counter.assign(0)
time_step = train_env.reset()
# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
returns = [avg_return]
driver = dynamic_episode_driver.DynamicEpisodeDriver(
    train_env,
    py_tf_eager_policy.PyTFEagerPolicy(
        tf_agent.collect_policy, use_tf_function=True),
    replay_observer,
    num_episodes=1)
iterator = iter(dataset)
for _ in range(num_iterations):

    # Collect a few episodes using collect_policy and save to the replay buffer.
    # time_step=train_env.reset()
    driver.run()

    # Use data from the buffer and update the agent's network.

    trajectories, unused_info = next(iterator)
    # batched_exp = tf.nest.map_structure(
    #     lambda t: tf.expand_dims(t, axis=0),
    #     trajectories
    # )

    train_loss = tf_agent.train(experience=trajectories)

    step = tf_agent.train_step_counter.numpy()

    if step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss.loss))

    if step % eval_interval == 0:
        avg_return = compute_avg_return(
            eval_env, tf_agent.policy, num_eval_episodes)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
        returns.append(avg_return)
