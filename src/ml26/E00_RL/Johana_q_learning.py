import gymnasium as gym
import numpy as np


class RandomAgent:
    def __init__(self, env, *args, **kwargs):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.num_actions = env.action_space.n

    def act(self, observation):
        return self.action_space.sample()

    def step(self, state, action, reward, next_state):
        pass


class QLearningAgent(RandomAgent):
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        super().__init__(env, alpha, gamma, epsilon)
        # Tabla estados x acciones
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))
        # Parameters
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate

    def act(self, observation):
        if np.random.random() < self.epsilon:
            return self.action_space.sample()  # Exploration
        else:
            return np.argmax(self.Q[observation])  # Exploitation

    # Update Q values using Q-learning
    def step(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.Q[next_state])
        # TODO: Implementa la actualización de Q-learning usando la ecuación vista en clase
        self.Q[state][action] = ...


if __name__ == "__main__":
    # TODO:
    # Este ejercicio cuenta como 5 pts extra en el primer examen parcial
    # 1. completa el código para implementar q learning,
    # 2. modifica los hiperparámetros para que el agente aprenda
    # 3. ejecuta el script para ver el comportamiento del agente
    # 4. Implementa una técnica para reducir la exploración conforme el agente aprende
    # https://gymnasium.farama.org/environments/toy_text/cliff_walking/
    env = gym.make("CliffWalking-v1", render_mode="human")

    n_episodes = 1000
    episode_length = 200
    agent = QLearningAgent(env, alpha=0.1, gamma=0.9, epsilon=0.9)
    for e in range(n_episodes):
        obs, _ = env.reset()
        ep_return = 0
        for i in range(episode_length):
            # take a random action
            action = agent.act(obs)
            next_obs, reward, done, _, _ = env.step(action)
            # update agent
            agent.step(obs, action, reward, next_obs)
            ep_return += reward
            obs = next_obs
            print(agent.Q)
            env.render()
            if done:
                break

        # TODO: Implementa algun código para reducir la exploración del agente conforme aprende
        # puedes decidir hacerlo por episodio, por paso del tiempo, retorno promedio, etc.

        print(f"Episode {e} return: ", ep_return)
    env.close()


"""
import gymnasium as gym
import numpy as np


class RandomAgent:
    def __init__(self, env, *args, **kwargs):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.num_actions = env.action_space.n

    def act(self, observation):
        return self.action_space.sample()

    def step(self, state, action, reward, next_state):
        pass


class QLearningAgent(RandomAgent):
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=1.0):
        super().__init__(env)

        # Tabla Q (estados × acciones)
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))

        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.epsilon = epsilon  # Exploration rate

    def act(self, observation):
        if np.random.random() < self.epsilon:
            return self.action_space.sample()  # Exploración
        else:
            return np.argmax(self.Q[observation])  # Explotación

    def step(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.Q[next_state])

        #  Q-learning update
        td_target = reward + self.gamma * self.Q[next_state][best_next_action]
        td_error = td_target - self.Q[state][action]

        self.Q[state][action] += self.alpha * td_error


if __name__ == "__main__":
    env = gym.make("CliffWalking-v1", render_mode="human")

    n_episodes = 500
    episode_length = 200

    agent = QLearningAgent(env, alpha=0.1, gamma=0.95, epsilon=1.0)

    epsilon_decay = 0.995
    epsilon_min = 0.05

    for e in range(n_episodes):
        obs, _ = env.reset()
        ep_return = 0

        for i in range(episode_length):
            action = agent.act(obs)

            next_obs, reward, done, truncated, _ = env.step(action)

            agent.step(obs, action, reward, next_obs)

            ep_return += reward
            obs = next_obs

            if done or truncated:
                break

        #  Reducir exploración
        agent.epsilon = max(epsilon_min, agent.epsilon * epsilon_decay)

        print(
            f"Episode {e} return: {ep_return:.2f}, epsilon: {agent.epsilon:.3f}")

    env.close()
"""
