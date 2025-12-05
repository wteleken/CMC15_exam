"""
Script de avaliação para visualizar agentes treinados.

Este script carrega as Q-tables treinadas e executa episódios com renderização
para visualizar o comportamento dos agentes.
"""

import pickle
import numpy as np
from environment import create_environment, create_bins, discretize_state


def evaluate_agent(q_table_path, episodes=5, agent_name="Agent"):
    """
    Avalia um agente treinado visualmente.
    
    Args:
        q_table_path (str): Caminho para o arquivo .pkl da Q-table
        episodes (int): Número de episódios para avaliar
        agent_name (str): Nome do agente para display
    """
    # Carrega Q-table treinada
    with open(q_table_path, 'rb') as f:
        q_table = pickle.load(f)
    
    print(f"\n{'='*60}")
    print(f"AVALIANDO: {agent_name}")
    print(f"{'='*60}")
    
    # Cria ambiente com visualização
    env = create_environment(render_mode='human')
    bins = create_bins()
    
    episode_rewards = []
    
    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            # Política 100% greedy (sem exploração)
            state = discretize_state(obs, bins)
            action = np.argmax(q_table[state])
            
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
        
        episode_rewards.append(total_reward)
        print(f"  Episódio {episode+1}: {total_reward:.0f} timesteps")
    
    env.close()
    
    # Estatísticas
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    print(f"\nEstatísticas ({episodes} episódios):")
    print(f"  Média: {mean_reward:.2f}")
    print(f"  Desvio padrão: {std_reward:.2f}")
    print(f"  Mínimo: {np.min(episode_rewards):.0f}")
    print(f"  Máximo: {np.max(episode_rewards):.0f}")
    
    return episode_rewards


def main():
    """Avalia ambos os agentes treinados."""
    
    print("\n" + "="*60)
    print("AVALIAÇÃO DE AGENTES TREINADOS")
    print("="*60)
    print("\nEste script executa episódios com os agentes treinados")
    print("usando política 100% greedy (sem exploração).\n")
    
    try:
        # Avalia Q-Learning
        qlearning_rewards = evaluate_agent(
            'qlearning_qtable.pkl', 
            episodes=5, 
            agent_name="Q-Learning"
        )
        
        print("\n" + "-"*60)
        
        # Avalia SARSA
        sarsa_rewards = evaluate_agent(
            'sarsa_qtable.pkl', 
            episodes=5, 
            agent_name="SARSA"
        )
        
        # Comparação final
        print("\n" + "="*60)
        print("COMPARAÇÃO FINAL")
        print("="*60)
        print(f"Q-Learning - Média: {np.mean(qlearning_rewards):.2f}")
        print(f"SARSA      - Média: {np.mean(sarsa_rewards):.2f}")
        
        if np.mean(qlearning_rewards) > np.mean(sarsa_rewards):
            print("\n🏆 Q-Learning teve melhor performance!")
        elif np.mean(qlearning_rewards) < np.mean(sarsa_rewards):
            print("\n🏆 SARSA teve melhor performance!")
        else:
            print("\n🤝 Empate técnico!")
        
    except FileNotFoundError as e:
        print(f"\n❌ Erro: {e}")
        print("\nExecute 'python comparison.py' primeiro para treinar os agentes.")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
