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


class SarsaLambdaAgent(BaseAgent):
    """
    Agente SARSA(λ) com Eligibility Traces (Accumulating Traces).
    
    Extensão do SARSA que propaga crédito de recompensas para estados anteriores
    através de "vestígios de elegibilidade" (eligibility traces).
    
    Atualização:
    1. Erro TD: δ = r + γ * Q(s',a') - Q(s,a)
    2. Incrementa vestígio: e(s,a) += 1
    3. Para todos os estados: Q(s,a) += α * δ * e(s,a)
    4. Decai vestígios: e(s,a) *= γ * λ
    
    Parâmetros:
    - λ = 0: SARSA tradicional (apenas estado atual)
    - λ = 1: Monte Carlo (crédito para todos os estados do episódio)
    - 0 < λ < 1: Balanço entre TD e MC (típico: 0.8-0.95)
    
    Referência: Sutton & Barto, Cap. 12 - Eligibility Traces
    """
    
    def __init__(self, state_shape, n_actions, alpha=0.1, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 lambda_factor=0.9, trace_threshold=0.001):
        """
        Inicializa o agente SARSA(λ).
        
        Args:
            state_shape (tuple): Forma do espaço de estados discreto
            n_actions (int): Número de ações possíveis
            alpha (float): Taxa de aprendizado
            gamma (float): Fator de desconto
            epsilon (float): Taxa inicial de exploração
            epsilon_decay (float): Fator de decaimento do epsilon
            epsilon_min (float): Valor mínimo de epsilon
            lambda_factor (float): Fator λ de decay dos traces (0 a 1)
            trace_threshold (float): Limiar para zerar traces insignificantes
        """
        super().__init__(state_shape, n_actions, alpha, gamma, 
                        epsilon, epsilon_decay, epsilon_min)
        
        self.lambda_factor = lambda_factor
        self.trace_threshold = trace_threshold
        
        # Tabela de vestígios de elegibilidade (mesma forma da Q-table)
        # Inicializada com zeros no início
        e_shape = state_shape + (n_actions,)
        self.e_table = np.zeros(e_shape)
    
    def reset_traces(self):
        """
        Zera a tabela de vestígios no início de cada episódio.
        
        Deve ser chamado no início de cada novo episódio para garantir
        que vestígios do episódio anterior não influenciem o atual.
        """
        self.e_table.fill(0.0)
    
    def update(self, state, action, reward, next_state, next_action):
        """
        Atualiza Q-table usando SARSA(λ) com Accumulating Traces.
        
        Args:
            state (tuple): Estado atual
            action (int): Ação tomada
            reward (float): Recompensa recebida
            next_state (tuple): Próximo estado
            next_action (int): Próxima ação real que será tomada (OBRIGATÓRIO)
        """
        if next_action is None:
            raise ValueError("SARSA(λ) requer next_action para atualização!")
        
        # 1. Calcula erro TD (temporal difference)
        current_q = self.q_table[state][action]
        next_q = self.q_table[next_state][next_action]
        delta = reward + self.gamma * next_q - current_q
        
        # 2. Incrementa vestígio do par estado-ação atual (Accumulating Traces)
        self.e_table[state][action] += 1.0
        
        # 3. Atualiza Q-table para TODOS os estados com vestígios ativos
        # Versão otimizada: apenas onde e_table > threshold
        active_traces = self.e_table > self.trace_threshold
        
        # Atualização vetorizada para eficiência
        self.q_table[active_traces] += self.alpha * delta * self.e_table[active_traces]
        
        # 4. Decai todos os vestígios (propaga influência para estados anteriores)
        self.e_table *= self.gamma * self.lambda_factor
        
        # Limpa vestígios insignificantes para economizar memória/computação
        self.e_table[self.e_table < self.trace_threshold] = 0.0
