"""
Módulo de treinamento para agentes de Reinforcement Learning.

Implementa funções de treino genéricas que suportam tanto Q-Learning quanto SARSA,
com lógica específica para cada algoritmo.
"""

from environment import create_environment, discretize_state


def train_agent(agent, bins, episodes=5000, is_sarsa=False, verbose=True):
    """
    Treina um agente de RL no ambiente CartPole.
    
    Args:
        agent: Agente a ser treinado (QLearningAgent ou SarsaAgent)
        bins (tuple): Bins de discretização (retorno de create_bins())
        episodes (int): Número de episódios de treinamento
        is_sarsa (bool): Se True, usa lógica SARSA; se False, usa Q-Learning
        verbose (bool): Se True, imprime progresso a cada 500 episódios
    
    Returns:
        list: Lista com a soma de recompensas de cada episódio
    """
    # Cria ambiente sem renderização para velocidade
    env = create_environment(render_mode=None, theta_threshold_deg=45)
    
    # Lista para armazenar recompensas de cada episódio
    episode_rewards = []
    
    for episode in range(episodes):
        # Reset do ambiente
        obs, info = env.reset()
        state = discretize_state(obs, bins)
        
        # Seleciona primeira ação
        action = agent.select_action(state)
        
        # Para SARSA, precisamos da próxima ação antes de atualizar
        if is_sarsa:
            next_action = None  # Será definido no loop
        
        episode_reward = 0
        done = False
        
        while not done:
            # Executa ação no ambiente
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Discretiza próximo estado
            next_state = discretize_state(obs, bins)
            
            # Acumula recompensa
            episode_reward += reward
            
            if is_sarsa:
                # SARSA: Seleciona próxima ação ANTES de atualizar
                if not done:
                    next_action = agent.select_action(next_state)
                else:
                    # No estado terminal, não há próxima ação
                    # Usamos ação dummy (0) mas o valor Q será 0 de qualquer forma
                    next_action = 0
                
                # Atualiza Q-table com a próxima ação real
                agent.update(state, action, reward, next_state, next_action)
                
                # Avança para próximo passo usando a ação já selecionada
                state = next_state
                action = next_action
            else:
                # Q-Learning: Atualiza sem precisar da próxima ação
                agent.update(state, action, reward, next_state)
                
                # Avança para próximo passo
                state = next_state
                if not done:
                    action = agent.select_action(state)
        
        # Armazena recompensa total do episódio
        episode_rewards.append(episode_reward)
        
        # Aplica decay do epsilon
        agent.decay_epsilon()
        
        # Log de progresso
        if verbose and (episode + 1) % 500 == 0:
            avg_reward = sum(episode_rewards[-100:]) / min(100, len(episode_rewards))
            print(f"Episódio {episode + 1}/{episodes} - "
                  f"Recompensa média (últimos 100): {avg_reward:.2f} - "
                  f"Epsilon: {agent.epsilon:.4f}")
    
    env.close()
    return episode_rewards
