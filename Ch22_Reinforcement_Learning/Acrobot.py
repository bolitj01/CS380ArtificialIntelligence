from collections import defaultdict
import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import pickle
import time


# Acrobot observation bounds from Gymnasium docs (v1):
# 0: cos(theta1) in [-1, 1]
# 1: sin(theta1) in [-1, 1]
# 2: cos(theta2) in [-1, 1]
# 3: sin(theta2) in [-1, 1]
# 4: thetaDot1 in [-4*pi, 4*pi]
# 5: thetaDot2 in [-9*pi, 9*pi]
STATE_BOUNDS = np.array([
    [-1.0, 1.0],
    [-1.0, 1.0],
    [-1.0, 1.0],
    [-1.0, 1.0],
    [-4.0 * np.pi, 4.0 * np.pi],
    [-9.0 * np.pi, 9.0 * np.pi],
], dtype=np.float32)

STATE_BINS = (8, 8, 8, 8, 12, 12)


def state_key(obs) -> tuple[int, int, int, int, int, int]:
    """Map continuous Acrobot observation to a discrete key for tabular Q-learning."""
    arr = np.asarray(obs, dtype=np.float32).reshape(-1)

    if arr.size != 6:
        raise ValueError(f"Expected Acrobot observation with 6 floats, got shape {arr.shape}")

    clipped = np.clip(arr, STATE_BOUNDS[:, 0], STATE_BOUNDS[:, 1])
    key_parts = []

    for i, bins in enumerate(STATE_BINS):
        low, high = STATE_BOUNDS[i]
        scaled = (clipped[i] - low) / (high - low)
        idx = int(np.floor(scaled * bins))
        idx = min(max(idx, 0), bins - 1)
        key_parts.append(idx)

    return tuple(key_parts)


################################### AGENT DEFINITION ###############################
class AcrobotAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.99,
    ):
        """Initialize a Q-learning agent for Acrobot-v1."""
        self.env = env

        # Q-table: maps (discrete_state, action) to expected return.
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

        print(f"Created {self.__class__.__name__}")

    def get_action(self, obs) -> int:
        """
        Choose an action using epsilon-greedy strategy.
        0 = apply -1 torque, 1 = apply 0 torque, 2 = apply +1 torque
        """
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return int(np.argmax(self.q_values[state_key(obs)]))

    def update(
        self,
        obs,
        action: int,
        reward: float,
        terminated: bool,
        next_obs,
    ):
        """Update Q-value from one transition."""
        obs_key = state_key(obs)
        next_obs_key = state_key(next_obs)

        future_q_value = (not terminated) * np.max(self.q_values[next_obs_key])
        target = reward + self.discount_factor * future_q_value
        temporal_difference = target - self.q_values[obs_key][action]

        self.q_values[obs_key][action] = (
            self.q_values[obs_key][action] + self.lr * temporal_difference
        )

        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


############################### TRAINING TIME! ###############################
def train_agent(agent, env, n_episodes=10, max_seconds=30):
    start_time = time.time()
    episodes_completed = 0

    for episode in tqdm(range(n_episodes)):
        if time.time() - start_time >= max_seconds:
            break

        obs, info = env.reset()
        done = False

        while not done:
            if time.time() - start_time >= max_seconds:
                done = True
                break

            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            agent.update(obs, action, reward, terminated, next_obs)

            done = terminated or truncated
            obs = next_obs

        agent.decay_epsilon()
        episodes_completed += 1

    elapsed = time.time() - start_time
    print(f"Training stopped after {elapsed:.2f}s and {episodes_completed} completed episodes.")

    return agent, env


########################## VISUALIZE LEARNING PROGRESS ###############################
def get_moving_avgs(arr, window, convolution_mode):
    """Compute moving average to smooth noisy data."""
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window


def visualize_learning(agent, env):
    # Smooth over up to a 500-episode window while preserving plotted length.
    rolling_length = max(1, min(500, len(env.return_queue)))
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

    # Episode rewards (win/loss performance)
    axs[0].set_title("Episode rewards")
    reward_moving_average = get_moving_avgs(
        env.return_queue,
        rolling_length,
        "valid"
    )
    axs[0].plot(range(rolling_length - 1, len(env.return_queue)), reward_moving_average)
    axs[0].set_ylabel("Average Reward")
    axs[0].set_xlabel("Episode")

    # Episode lengths (how many actions per hand)
    axs[1].set_title("Episode lengths")
    length_moving_average = get_moving_avgs(
        env.length_queue,
        rolling_length,
        "valid"
    )
    axs[1].plot(range(rolling_length - 1, len(env.length_queue)), length_moving_average)
    axs[1].set_ylabel("Average Episode Length")
    axs[1].set_xlabel("Episode")

    # Training error (how much we're still learning)
    axs[2].set_title("Training Error")
    training_error_moving_average = get_moving_avgs(
        agent.training_error,
        rolling_length,
        "valid"
    )
    axs[2].plot(range(rolling_length - 1, len(agent.training_error)), training_error_moving_average)
    axs[2].set_ylabel("Temporal Difference Error")
    axs[2].set_xlabel("Step")

    plt.tight_layout()
    plt.show()


######################## EVALUATE THE LEARNED POLICY #####################
def test_agent(agent, env, n_episodes=3):
    """Test agent performance without learning or exploration."""
    total_rewards = []

    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    for _ in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        total_rewards.append(episode_reward)

    agent.epsilon = old_epsilon

    average_reward = np.mean(total_rewards)

    print(f"Test Results over {n_episodes} episodes:")
    print(f"Average Reward: {average_reward:.3f}")
    print(f"Standard Deviation: {np.std(total_rewards):.3f}")


######################## SAVE THE TRAINED AGENT #####################
def save_agent(agent):
    print(f"Saving {len(agent.q_values)} Q-values...")
    with open("Ch22_Reinforcement_Learning/acrobot_q_values.pkl", "wb") as f:
        pickle.dump(dict(agent.q_values), f)
    print("Saved Q-values to acrobot_q_values.pkl")


def load_agent(agent):
    with open("Ch22_Reinforcement_Learning/acrobot_q_values.pkl", "rb") as f:
        loaded_q_values = pickle.load(f)
        agent.q_values = defaultdict(lambda: np.zeros(agent.env.action_space.n), loaded_q_values)
    print("Loaded pre-trained Q-values from acrobot_q_values.pkl")


######################## MAIN #####################
def main():
    # Hyperparameters for Acrobot's sparse-reward swing-up task.
    learning_rate = 0.15
    n_episodes = 600
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / n_episodes
    final_epsilon = 0.05

    # Training environment requested by user style.
    env = gym.make("Acrobot-v1", render_mode="rgb_array")
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

    agent = AcrobotAgent(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
    )

    # Optionally, load pre-trained Q-values for further training/testing.
    # load_agent(agent)

    train_agent(agent, env, n_episodes=n_episodes, max_seconds=30)
    visualize_learning(agent, env)

    # Human-rendered evaluation to watch torque-driven swing-up.
    eval_env = gym.make("Acrobot-v1", render_mode="human")
    test_agent(agent, eval_env, n_episodes=3)

    save_agent(agent)
    eval_env.close()
    env.close()


if __name__ == "__main__":
    main()