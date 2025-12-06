import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------
#   DISCRETIZAÇÃO DO ESTADO
# ------------------------------
def create_bins(n_bins=10):
    bins = [
        np.linspace(-4.8, 4.8, n_bins),        # posição do carrinho
        np.linspace(-3.0, 3.0, n_bins),        # velocidade
        np.linspace(-0.418, 0.418, n_bins),    # ângulo do pêndulo
        np.linspace(-3.0, 3.0, n_bins),        # velocidade angular
    ]
    return bins


def discretize_state(state, bins):
    return tuple(
        int(np.digitize(s, bins[i]) - 1)
        for i, s in enumerate(state)
    )


# ------------------------------
#   AGENTES
# ------------------------------

class QLearningAgent:
    def __init__(self, bins, n_actions, lr, gamma, epsilon):
        self.bins = bins
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

        shape = tuple(len(b) for b in bins) + (n_actions,)
        self.Q = np.zeros(shape)

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state, done):
        target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        self.Q[state][action] += self.lr * (target - self.Q[state][action])


class SarsaAgent:
    def __init__(self, bins, n_actions, lr, gamma, epsilon):
        self.bins = bins
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

        shape = tuple(len(b) for b in bins) + (n_actions,)
        self.Q = np.zeros(shape)

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state, next_action, done):
        target = reward + (0 if done else self.gamma * self.Q[next_state][next_action])
        self.Q[state][action] += self.lr * (target - self.Q[state][action])


# ------------------------------
#   TREINO DE UM AGENTE
# ------------------------------

def train_agent(agent, episodes=5000):
    env = gym.make("CartPole-v1")
    rewards = []
    bins = agent.bins

    for ep in range(episodes):
        state, _ = env.reset()
        state = discretize_state(state, bins)

        action = agent.choose_action(state) if isinstance(agent, SarsaAgent) else None

        total_reward = 0

        while True:
            next_state_raw, reward, terminated, truncated, _ = env.step(
                action if action is not None else agent.choose_action(state)
            )
            done = terminated or truncated

            next_state = discretize_state(next_state_raw, bins)
            total_reward += reward

            if isinstance(agent, QLearningAgent):
                next_action = None
                agent.update(state, agent.choose_action(state), reward, next_state, done)

            else:  # SARSA
                next_action = agent.choose_action(next_state)
                agent.update(state, action, reward, next_state, next_action, done)
                action = next_action

            state = next_state

            if done:
                break

        rewards.append(total_reward)

        # decaimento do epsilon
        agent.epsilon = max(0.01, agent.epsilon * 0.999)

    return rewards


# ------------------------------
#   MÉDIA MÓVEL PARA GRÁFICOS
# ------------------------------
def moving_average(x, n=100):
    return np.convolve(x, np.ones(n)/n, mode='valid')


# ------------------------------
#   EXECUÇÃO PRINCIPAL
# ------------------------------

if __name__ == "__main__":
    bins = create_bins(12)

    params = {
        "lr": 0.1,
        "gamma": 0.99,
        "epsilon": 1.0,
    }

    q_agent = QLearningAgent(bins, 2, **params)
    sarsa_agent = SarsaAgent(bins, 2, **params)

    print("Treinando Q-Learning...")
    rewards_q = train_agent(q_agent, episodes=5000)

    print("Treinando SARSA...")
    rewards_sarsa = train_agent(sarsa_agent, episodes=5000)

    # salvar dados
    np.save("rewards_q.npy", rewards_q)
    np.save("rewards_sarsa.npy", rewards_sarsa)

    # gráfico
    plt.figure(figsize=(10, 5))
    plt.plot(moving_average(rewards_q), label="Q-Learning")
    plt.plot(moving_average(rewards_sarsa), label="SARSA")
    plt.xlabel("Episódios")
    plt.ylabel("Recompensa Média (window=100)")
    plt.title("Comparação Q-Learning vs SARSA — CartPole")
    plt.legend()
    plt.grid()
    plt.show()

    print("\nArquivos salvos:")
    print(" → rewards_q.npy")
    print(" → rewards_sarsa.npy")
