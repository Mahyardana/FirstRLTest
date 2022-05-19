from __future__ import absolute_import, division, print_function


import os
import gym
import base64
import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import pyvirtualdisplay
from IPython import display
from matplotlib import pyplot as PLT

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.environments import batched_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents import specs
from tf_agents.drivers import dynamic_step_driver
from tf_agents.networks import q_network
from tf_agents.replay_buffers import py_uniform_replay_buffer
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import time_step
from tf_agents.replay_buffers import table
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
imageio.plugins.ffmpeg.download()

num_iterations = 200000  # @param {type:"integer"}

initial_collect_steps = 100  # @param {type:"integer"}
collect_steps_per_iteration = 1  # @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}

batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
log_interval = 2000  # @param {type:"integer"}

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 10000  # @param {type:"integer"}

env_name = "CartPole-v0"
env = suite_gym.load(env_name)
env.reset()

train_py_env = suite_gym.load(env_name)
eval_py_env = suite_gym.load(env_name)

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

# os.system('pause')

fc_layer_params = (100, 50)
action_tensor_spec = tensor_spec.from_spec(env.action_spec())
num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1


# Define a helper function to create Dense layers configured with the right
# activation and kernel initializer.
def dense_layer(num_units):
    return tf.keras.layers.Dense(
        num_units,
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0, mode="fan_in", distribution="truncated_normal"
        ),
    )


# QNetwork consists of a sequence of Dense layers followed by a dense layer
# with `num_actions` units to generate one q_value per available action as
# its output.
dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
q_values_layer = tf.keras.layers.Dense(
    num_actions,
    activation=None,
    kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.03, maxval=0.03),
    bias_initializer=tf.keras.initializers.Constant(-0.2),
)
q_net = sequential.Sequential(dense_layers + [q_values_layer])

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)

agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter,
)

agent.initialize()


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


# See also the metrics module for standard implementations of different metrics.
# https://github.com/tensorflow/agents/tree/master/tf_agents/metrics

data_spec = agent.collect_data_spec
batch_size=train_env.batch_size
max_length = 100000


replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec, batch_size=batch_size, max_length=max_length
)

replay_observer = [replay_buffer.add_batch]

# env.reset()
# collect_op  = dynamic_step_driver.DynamicStepDriver(
#    env, policy, replay_observer, num_steps=2
# )


# Dataset generates trajectories with shape [Bx2x...]
dataset = replay_buffer.as_dataset(
    num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2
).prefetch(3)


# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# Reset the train step.
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
returns = [avg_return]

# Reset the environment.
time_step = train_env.reset()

policy = py_tf_eager_policy.PyTFEagerPolicy(agent.collect_policy, use_tf_function=True,batch_time_steps=False)

# collect_op = dynamic_step_driver.DynamicStepDriver(
#  train_env,
#  policy,
#  observers=replay_observer,
#  num_steps=1
# )
collect_driver = py_driver.PyDriver(
    train_env,
    policy,
    replay_observer,
    max_steps=collect_steps_per_iteration,
)

collect_driver.run(time_step)

iterator = iter(dataset)


def create_policy_eval_video(policy, filename, num_episodes=5, fps=30):
    filename = filename + ".mp4"
    with imageio.get_writer(filename, fps=fps) as video:
        for _ in range(num_episodes):
            time_step = eval_env.reset()
            video.append_data(eval_py_env.render())
            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = eval_env.step(action_step.action)
                video.append_data(eval_py_env.render())


for _ in range(num_iterations):
    # Collect a few steps and save to the replay buffer.
    time_step, _ = collect_driver.run(time_step)

    # Sample a batch of data from the buffer and update the agent's network.
    experience, unused_info = next(iterator)
    train_loss = agent.train(experience).loss

    step = agent.train_step_counter.numpy()
    if step % log_interval == 0:
        print("step = {0}: loss = {1}".format(step, train_loss))

    if step % eval_interval == 0:
        avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
        print("step = {0}: Average Return = {1}".format(step, avg_return))
        returns.append(avg_return)
        create_policy_eval_video(agent.policy, "trained-agent{}".format(step))


# def dense_layer(num_units):
#    return layers.Dense(
#        num_units,
#        activation=tf.keras.activations.relu,
#        kernel_initializer=tf.keras.initializers.VarianceScaling(
#            scale=2.0,mode='fan_in',distribution='truncated_normal'))

# dense_layers=[dense_layer(num_units) for num_units in fc_layer_params]
# q_values_layer=tf.keras.layers.Dense(
#    num_actions,
#    activation=None,
#    kernel_initializer=tf.keras.initializers.RandomUniform(
#        minval=-0.03,maxval=0.03),
#    bias_initializer=tf.keras.inititalizers.Constant(-0.2))
# q_net=sequential.Sequential(dense_layers+[q_values_layer])

# agent=dqn_agent.DqnAgent(
#    train_env.time_step_spec(),
#    train_env.action_spec,
#    q_network=q_net,
#    optimizer=optimizer,
#    td_errors_loss_fn=common.element_wise_squared_loss,
#    train_step_counter=train_step_counter)
# agent.initialize()

# eval_policy=agent.policy
# collect_policy=agent.collect_policy

# table_name='uniform_table'
# replay_buffer_signature=tensor_spec.from_spec(agent.collect_data_spec)
# replay_buffer_signature=tensor_spec.add_outer_dim(replay_buffer_signature)
# table=reverb.table(
#    table_name,
#    max_size=replay_buffer_max_length,
#    sampler=reverb.selectors.Uniform(),
#    remover=reverb.selectors.Fifo(),
#    rate_limiter=reverb.rate_limiters.MinSize(1),
#    signature=replay_buffer_signature)
# reverb_server=reverb.Server([table])

# replay_buffer=reverb_replay_buffer.ReverbReplayBuffer(
#    agent.collect_data_spec,
#    table_name=table_name,
#    sequence_length=2,
#    local_server=reverb_server)

# rb_observer=reverb_utils.ReverbAddTrajectoryObserver(
#    replay_buffer.py_client,
#    table_name,
#    sequence_length=2)

# collect_driver=py_driver.PyDriver(
#    env,
#    py_tf_eager_policy.PyTFEagerPolicy(
#        agent.collect_policy,user_tf_function=True),
#    [rb_observer],
#    max_steps=collect_steps_per_iteration)

# for _ in range(num_iterations):
#    time_step,_=collect_driver.run(time_step)

#    experience,unused_info=next(iterator)
#    train_loss=agent.train(experience).loss

#    step=agent.train_step_counter.numpy()

#    if step%log_interval==0:
#        print('step = {0}: loss = {1}'.format(step,train_loss))

#    if step%eval_interval==0:
#        avg_return=compute_avg_return(eval_env,agent.policy,num_eval_episodes)
#        print('step = {0}: Average Return = {1}'.format(step,avg_return))
#        returns.append(avg_return)


# start=time.perf_counter()
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
## Configuration parameters for the whole setup
# seed = 42
# gamma = 0.99  # Discount factor for past rewards
# max_steps_per_episode = 100000
# env = gym.make("CartPole-v1")  # Create the environment
# env.reset(seed=seed)
# eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0

# num_inputs = 4
# num_actions = 2
# num_hidden = 128

# inputs = layers.Input(shape=(num_inputs,))
# common = layers.Dense(num_hidden, activation="relu")(inputs)
# action = layers.Dense(num_actions, activation="softmax")(common)
# critic = layers.Dense(1)(common)

# model = keras.Model(inputs=inputs, outputs=[action, critic])

# optimizer = keras.optimizers.Adam(learning_rate=0.01)
# huber_loss = keras.losses.Huber()
# action_probs_history = []
# critic_value_history = []
# rewards_history = []
# running_reward = 0
# episode_count = 0

# while True:  # Run until solved
#    state = env.reset()
#    episode_reward = 0
#    with tf.GradientTape() as tape:
#        for timestep in range(1, max_steps_per_episode):
#            # env.render(); Adding this line would show the attempts
#            # of the agent in a pop up window.
#            env.render();
#            state = tf.convert_to_tensor(state)
#            state = tf.expand_dims(state, 0)

#            # Predict action probabilities and estimated future rewards
#            # from environment state
#            action_probs, critic_value = model(state)
#            critic_value_history.append(critic_value[0, 0])

#            # Sample action from action probability distribution
#            action = np.random.choice(num_actions, p=np.squeeze(action_probs))
#            action_probs_history.append(tf.math.log(action_probs[0, action]))

#            # Apply the sampled action in our environment
#            state, reward, done, _ = env.step(action)
#            rewards_history.append(reward)
#            episode_reward += reward

#            if done:
#                break

#        # Update running reward to check condition for solving
#        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

#        # Calculate expected value from rewards
#        # - At each timestep what was the total reward received after that timestep
#        # - Rewards in the past are discounted by multiplying them with gamma
#        # - These are the labels for our critic
#        returns = []
#        discounted_sum = 0
#        for r in rewards_history[::-1]:
#            discounted_sum = r + gamma * discounted_sum
#            returns.insert(0, discounted_sum)

#        # Normalize
#        returns = np.array(returns)
#        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
#        returns = returns.tolist()

#        # Calculating loss values to update our network
#        history = zip(action_probs_history, critic_value_history, returns)
#        actor_losses = []
#        critic_losses = []
#        for log_prob, value, ret in history:
#            # At this point in history, the critic estimated that we would get a
#            # total reward = `value` in the future. We took an action with log probability
#            # of `log_prob` and ended up recieving a total reward = `ret`.
#            # The actor must be updated so that it predicts an action that leads to
#            # high rewards (compared to critic's estimate) with high probability.
#            diff = ret - value
#            actor_losses.append(-log_prob * diff)  # actor loss

#            # The critic must be updated so that it predicts a better estimate of
#            # the future rewards.
#            critic_losses.append(
#                huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
#            )

#        # Backpropagation
#        loss_value = sum(actor_losses) + sum(critic_losses)
#        grads = tape.gradient(loss_value, model.trainable_variables)
#        optimizer.apply_gradients(zip(grads, model.trainable_variables))

#        # Clear the loss and reward history
#        action_probs_history.clear()
#        critic_value_history.clear()
#        rewards_history.clear()

#    # Log details
#    episode_count += 1
#    if episode_count % 10 == 0:
#        template = "running reward: {:.2f} at episode {}"
#        print(template.format(running_reward, episode_count))
#        print("TimePassed:{}".format(time.perf_counter()-start))

#    if running_reward > 195:  # Condition to consider the task solved
#        print("Solved at episode {}!".format(episode_count))
#        break
