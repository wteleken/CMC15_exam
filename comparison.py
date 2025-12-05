"""
Script principal para comparação entre Q-Learning e SARSA no CartPole.

Treina ambos os algoritmos com os mesmos hiperparâmetros e gera gráfico comparativo
das curvas de aprendizado.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from environment import create_bins
from agents import QLearningAgent, SarsaAgent
from train import train_agent


def moving_average(data, window_size=100):
    """
    Calcula média móvel para suavizar curvas.
    
    Args:
        data (list): Dados originais
        window_size (int): Tamanho da janela
    
    Returns:
        np.array: Dados suavizados
    """
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')


def main():
    """Função principal que executa o treino e comparação."""
    
    print("=" * 70)
    print("COMPARAÇÃO: Q-LEARNING vs SARSA - CartPole-v1")
    print("=" * 70)
    
    # Configuração de seeds para reprodutibilidade
    np.random.seed(42)
    
    # Hiperparâmetros fixos (conforme especificação)
    ALPHA = 0.1
    GAMMA = 0.99
    EPSILON = 1.0
    EPSILON_DECAY = 0.995
    EPSILON_MIN = 0.01
    EPISODES = 5000
    
    # Configuração do espaço de estados discreto
    STATE_SHAPE = (6, 6, 10, 10)  # (x, x_dot, theta, theta_dot)
    N_ACTIONS = 2  # Esquerda ou Direita
    
    # Cria bins de discretização
    bins = create_bins()
    
    print(f"\nHiperparâmetros:")
    print(f"  - Alpha (taxa de aprendizado): {ALPHA}")
    print(f"  - Gamma (fator de desconto): {GAMMA}")
    print(f"  - Epsilon inicial: {EPSILON}")
    print(f"  - Epsilon decay: {EPSILON_DECAY}")
    print(f"  - Epsilon mínimo: {EPSILON_MIN}")
    print(f"  - Episódios de treinamento: {EPISODES}")
    print(f"  - Espaço de estados: {STATE_SHAPE} = {np.prod(STATE_SHAPE)} estados")
    print(f"  - Ações: {N_ACTIONS}")
    
    # ========================================================================
    # TREINAMENTO Q-LEARNING
    # ========================================================================
    print("\n" + "=" * 70)
    print("TREINANDO Q-LEARNING (Off-policy)")
    print("=" * 70)
    
    qlearning_agent = QLearningAgent(
        state_shape=STATE_SHAPE,
        n_actions=N_ACTIONS,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon=EPSILON,
        epsilon_decay=EPSILON_DECAY,
        epsilon_min=EPSILON_MIN
    )
    
    qlearning_rewards = train_agent(
        agent=qlearning_agent,
        bins=bins,
        episodes=EPISODES,
        is_sarsa=False,
        verbose=True
    )
    
    # Salva Q-table treinada
    with open('qlearning_qtable.pkl', 'wb') as f:
        pickle.dump(qlearning_agent.q_table, f)
    print("\nQ-table do Q-Learning salva em 'qlearning_qtable.pkl'")
    
    # ========================================================================
    # TREINAMENTO SARSA
    # ========================================================================
    print("\n" + "=" * 70)
    print("TREINANDO SARSA (On-policy)")
    print("=" * 70)
    
    # Reset da seed para comparação justa
    np.random.seed(42)
    
    sarsa_agent = SarsaAgent(
        state_shape=STATE_SHAPE,
        n_actions=N_ACTIONS,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon=EPSILON,
        epsilon_decay=EPSILON_DECAY,
        epsilon_min=EPSILON_MIN
    )
    
    sarsa_rewards = train_agent(
        agent=sarsa_agent,
        bins=bins,
        episodes=EPISODES,
        is_sarsa=True,
        verbose=True
    )
    
    # Salva Q-table treinada
    with open('sarsa_qtable.pkl', 'wb') as f:
        pickle.dump(sarsa_agent.q_table, f)
    print("\nQ-table do SARSA salva em 'sarsa_qtable.pkl'")
    
    # ========================================================================
    # ANÁLISE E COMPARAÇÃO
    # ========================================================================
    print("\n" + "=" * 70)
    print("ANÁLISE DE RESULTADOS")
    print("=" * 70)
    
    # Estatísticas finais (últimos 100 episódios)
    qlearning_final = np.mean(qlearning_rewards[-100:])
    sarsa_final = np.mean(sarsa_rewards[-100:])
    
    qlearning_max = np.max(qlearning_rewards)
    sarsa_max = np.max(sarsa_rewards)
    
    print(f"\nQ-Learning:")
    print(f"  - Recompensa média (últimos 100 episódios): {qlearning_final:.2f}")
    print(f"  - Recompensa máxima: {qlearning_max:.2f}")
    print(f"  - Desvio padrão (últimos 100): {np.std(qlearning_rewards[-100:]):.2f}")
    
    print(f"\nSARSA:")
    print(f"  - Recompensa média (últimos 100 episódios): {sarsa_final:.2f}")
    print(f"  - Recompensa máxima: {sarsa_max:.2f}")
    print(f"  - Desvio padrão (últimos 100): {np.std(sarsa_rewards[-100:]):.2f}")
    
    # ========================================================================
    # GERAÇÃO DO GRÁFICO COMPARATIVO
    # ========================================================================
    print("\n" + "=" * 70)
    print("GERANDO GRÁFICO COMPARATIVO")
    print("=" * 70)
    
    # Calcula médias móveis
    window = 100
    qlearning_smooth = moving_average(qlearning_rewards, window)
    sarsa_smooth = moving_average(sarsa_rewards, window)
    
    # Cria figura
    plt.figure(figsize=(12, 6))
    
    # Plot das curvas suavizadas
    episodes_range = range(window - 1, EPISODES)
    plt.plot(episodes_range, qlearning_smooth, label='Q-Learning (Off-policy)', 
             linewidth=2, color='blue', alpha=0.8)
    plt.plot(episodes_range, sarsa_smooth, label='SARSA (On-policy)', 
             linewidth=2, color='red', alpha=0.8)
    
    # Configurações do gráfico
    plt.xlabel('Episódio', fontsize=12)
    plt.ylabel('Recompensa Total (Média Móvel)', fontsize=12)
    plt.title('Comparação: Q-Learning vs SARSA - CartPole-v1\n' + 
              f'(Média móvel de {window} episódios)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Salva figura
    plt.savefig('comparison_result.png', dpi=300, bbox_inches='tight')
    print("Gráfico salvo como 'comparison_result.png'")
    
    # Mostra gráfico
    plt.show()
    
    print("\n" + "=" * 70)
    print("COMPARAÇÃO CONCLUÍDA!")
    print("=" * 70)
    print("\nArquivos gerados:")
    print("  - qlearning_qtable.pkl")
    print("  - sarsa_qtable.pkl")
    print("  - comparison_result.png")
    print("\nPara visualizar os agentes treinados, considere criar um script")
    print("de avaliação que carregue as Q-tables e execute com render_mode='human'.")


if __name__ == "__main__":
    main()
