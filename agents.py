"""
Módulo de agentes para Reinforcement Learning tabular.

Implementa Q-Learning (off-policy) e SARSA (on-policy) para o problema CartPole
com discretização do espaço de estados.
"""

import numpy as np
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """
    Classe base abstrata para agentes de RL tabular.
    
    Implementa Q-table, política epsilon-greedy e estrutura comum.
    Subclasses devem implementar o método update().
    """
    
    def __init__(self, state_shape, n_actions, alpha=0.1, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        """
        Inicializa o agente base.
        
        Args:
            state_shape (tuple): Forma do espaço de estados discreto (ex: (6,6,10,10))
            n_actions (int): Número de ações possíveis
            alpha (float): Taxa de aprendizado
            gamma (float): Fator de desconto
            epsilon (float): Taxa inicial de exploração
            epsilon_decay (float): Fator de decaimento do epsilon
            epsilon_min (float): Valor mínimo de epsilon
        """
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Inicializa Q-table - otimista para SARSA, zeros para Q-Learning
        # Shape: (*state_shape, n_actions) = (8, 8, 12, 12, 2)
        q_shape = state_shape + (n_actions,)
        self.q_table = np.zeros(q_shape)
        
        # Contador de visitas para UCB exploration (melhor exploração de estados)
        self.visit_counts = np.zeros(q_shape)
        self.total_steps = 0
    
    def select_action_ucb(self, state, c=2.0):
        """
        Seleciona ação usando Upper Confidence Bound para melhor exploração.
        
        Args:
            state (tuple): Estado discretizado
            c (float): Constante de exploração (maior = mais exploração)
        
        Returns:
            int: Ação escolhida
        """
        if self.total_steps == 0:
            return np.random.randint(self.n_actions)
        
        q_values = self.q_table[state]
        visit_counts = self.visit_counts[state]
        
        # Bonus de exploração: c * sqrt(ln(total) / visits)
        # Estados não visitados recebem bonus infinito
        bonus = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            if visit_counts[a] == 0:
                bonus[a] = float('inf')  # Prioriza estados nunca visitados
            else:
                bonus[a] = c * np.sqrt(np.log(self.total_steps) / visit_counts[a])
        
        ucb_values = q_values + bonus
        return np.argmax(ucb_values)
    
    def select_action(self, state):
        """
        Seleciona ação usando política epsilon-greedy.
        
        Args:
            state (tuple): Estado discreto (x_idx, x_dot_idx, theta_idx, theta_dot_idx)
        
        Returns:
            int: Ação selecionada (0 ou 1)
        """
        self.total_steps += 1
        
        # Exploração: ação aleatória
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            # Exploração: melhor ação conhecida (greedy)
            action = np.argmax(self.q_table[state])
        
        # Atualiza contador de visitas
        self.visit_counts[state][action] += 1
        return action
    
    def decay_epsilon(self):
        """Aplica decaimento ao epsilon após cada episódio."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    @abstractmethod
    def update(self, state, action, reward, next_state, next_action=None):
        """
        Atualiza a Q-table (método abstrato).
        
        Args:
            state (tuple): Estado atual
            action (int): Ação tomada
            reward (float): Recompensa recebida
            next_state (tuple): Próximo estado
            next_action (int, optional): Próxima ação (necessário para SARSA)
        """
        pass
    
    def get_q_value(self, state, action):
        """Retorna o valor Q para um par estado-ação."""
        return self.q_table[state][action]


class QLearningAgent(BaseAgent):
    """
    Agente Q-Learning (off-policy).
    
    Atualização baseada na MELHOR ação futura possível:
    Q(s,a) += α * (r + γ * max_a'(Q(s',a')) - Q(s,a))
    
    Referência: Sutton & Barto, Reinforcement Learning: An Introduction
    """
    
    def update(self, state, action, reward, next_state, next_action=None):
        """
        Atualiza Q-table usando regra do Q-Learning.
        
        Args:
            state (tuple): Estado atual
            action (int): Ação tomada
            reward (float): Recompensa recebida
            next_state (tuple): Próximo estado
            next_action (int, optional): Ignorado no Q-Learning
        """
        # Valor atual
        current_q = self.q_table[state][action]
        
        # Melhor valor futuro possível (max sobre todas as ações)
        max_next_q = np.max(self.q_table[next_state])
        
        # Atualização Q-Learning
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        
        self.q_table[state][action] = new_q


class SarsaAgent(BaseAgent):
    """
    Agente SARSA (on-policy).
    
    Atualização baseada na PRÓXIMA ação real escolhida:
    Q(s,a) += α * (r + γ * Q(s',a') - Q(s,a))
    
    Diferença crucial: usa a ação que será realmente tomada (a'), não a melhor.
    
    Referência: Sutton & Barto, Reinforcement Learning: An Introduction
    """
    
    def update(self, state, action, reward, next_state, next_action):
        """
        Atualiza Q-table usando regra do SARSA.
        
        Args:
            state (tuple): Estado atual
            action (int): Ação tomada
            reward (float): Recompensa recebida
            next_state (tuple): Próximo estado
            next_action (int): Próxima ação real que será tomada (OBRIGATÓRIO)
        """
        if next_action is None:
            raise ValueError("SARSA requer next_action para atualização!")
        
        # Valor atual
        current_q = self.q_table[state][action]
        
        # Valor da próxima ação REAL (não o máximo)
        next_q = self.q_table[next_state][next_action]
        
        # Atualização SARSA
        new_q = current_q + self.alpha * (reward + self.gamma * next_q - current_q)
        
        self.q_table[state][action] = new_q
