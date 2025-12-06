"""
Módulo de treinamento para agentes de Reinforcement Learning.

Implementa funções de treino genéricas que suportam tanto Q-Learning quanto SARSA,
com lógica específica para cada algoritmo.
"""

from environment import create_environment, discretize_state


def train_agent(agent, bins, episodes=5000, is_sarsa=False, use_reward_shaping=True, 
                use_ucb=False, alpha_start=None, alpha_end=None, verbose=True):
    """
    Treina um agente de RL no ambiente CartPole.
    
    Args:
        agent: Agente a ser treinado (QLearningAgent ou SarsaAgent)
        bins (tuple): Bins de discretização (retorno de create_bins())
        episodes (int): Número de episódios de treinamento
        is_sarsa (bool): Se True, usa lógica SARSA; se False, usa Q-Learning
        use_reward_shaping (bool): Se True, adiciona reward shaping (FASE 1B)
        use_ucb (bool): Se True, usa UCB exploration (melhor para exploração de estados)
        alpha_start (float): Alpha inicial para decay adaptativo (None = fixo)
        alpha_end (float): Alpha final para decay adaptativo
        verbose (bool): Se True, imprime progresso a cada 500 episódios
    
    Returns:
        list: Lista com a soma de recompensas de cada episódio
    """
    # Cria ambiente sem renderização para velocidade
    env = create_environment(render_mode=None, theta_threshold_deg=45)
    
    # Lista para armazenar recompensas de cada episódio
    episode_rewards = []
    
    for episode in range(episodes):
        # Alpha adaptativo (se configurado)
        if alpha_start is not None and alpha_end is not None:
            # Decay exponencial do alpha
            alpha_decay_rate = (alpha_end / alpha_start) ** (1 / episodes)
            agent.alpha = max(alpha_end, alpha_start * (alpha_decay_rate ** episode))
        
        # Reset do ambiente
        obs, info = env.reset()
        state = discretize_state(obs, bins)
        
        # Seleciona primeira ação (UCB para primeiros 30% dos episódios se habilitado)
        if use_ucb and episode < int(0.3 * episodes):
            action = agent.select_action_ucb(state, c=2.0)
        else:
            action = agent.select_action(state)
        
        # Para SARSA, precisamos da próxima ação antes de atualizar
        if is_sarsa:
            next_action = None  # Será definido no loop
        
        episode_reward = 0
        done = False
        timesteps = 0
        
        while not done:
            # Executa ação no ambiente
            obs, original_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Conta timesteps (reward original do CartPole é sempre 1.0 por step)
            timesteps += 1
            
            # FASE 1B: Reward Shaping (apenas para aprendizado, não para métrica)
            reward = original_reward
            if use_reward_shaping:
                cart_pos, cart_vel, pole_angle, pole_vel = obs
                
                # Bônus de estabilidade (quanto mais próximo de vertical, melhor)
                # theta_threshold = 0.209 rad (12 graus)
                stability_bonus = 0.2 * (1.0 - abs(pole_angle) / 0.209)
                
                # Penalidade de velocidade (penaliza movimentos bruscos)
                velocity_penalty = -0.05 * (abs(cart_vel) + abs(pole_vel))
                
                # Recompensa moldada (para aprendizado)
                reward = original_reward + stability_bonus + velocity_penalty
            
            # Discretiza próximo estado
            next_state = discretize_state(obs, bins)
            
            # Acumula timesteps como métrica de episódio
            episode_reward = timesteps
            
            if is_sarsa:
                # SARSA: Seleciona próxima ação ANTES de atualizar
                if not done:
                    # Usa UCB nos primeiros 30% dos episódios se habilitado
                    if use_ucb and episode < int(0.3 * episodes):
                        next_action = agent.select_action_ucb(next_state, c=2.0)
                    else:
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
                    # Q-Learning não usa UCB (já é off-policy e eficiente)
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
