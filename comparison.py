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
    
    # Hiperparâmetros otimizados
    # Q-Learning: config conservadora comprovada
    # SARSA: config agressiva para maximizar exploração e convergência
    
    # Comum para ambos
    GAMMA = 0.99
    EPSILON = 1.0
    N_ACTIONS = 2
    STATE_SHAPE = (8, 8, 12, 12)  # 9216 estados
    
    # Específico Q-Learning (mantém o que funcionou)
    QL_ALPHA = 0.15
    QL_EPSILON_DECAY = 0.9995
    QL_EPSILON_MIN = 0.001
    QL_EPISODES = 25000
    
    # Específico SARSA (otimizado para exploração + convergência)
    SARSA_ALPHA_START = 0.3      # Aprende rápido no início
    SARSA_ALPHA_END = 0.05       # Refina no final
    SARSA_EPSILON_DECAY = 0.9993 # Decay mais rápido (reduz exploração aleatória)
    SARSA_EPSILON_MIN = 0.0001   # Ultra-baixo para convergência determinística
    SARSA_EPISODES = 50000       # Dobro do tempo para convergência completa
    SARSA_USE_UCB = True         # Usa UCB para melhor exploração de estados
    
    # Cria bins de discretização
    bins = create_bins(STATE_SHAPE)
    
    print(f"\nConfiguração do Ambiente:")
    print(f"  - Espaço de estados: {STATE_SHAPE} = {np.prod(STATE_SHAPE)} estados")
    print(f"  - Ações: {N_ACTIONS}")
    print(f"  - Gamma: {GAMMA}")
    
    print(f"\nQ-Learning (config conservadora):")
    print(f"  - Alpha: {QL_ALPHA}")
    print(f"  - Epsilon decay: {QL_EPSILON_DECAY}, min: {QL_EPSILON_MIN}")
    print(f"  - Episódios: {QL_EPISODES}")
    
    print(f"\nSARSA (config otimizada para exploração):")
    print(f"  - Alpha: {SARSA_ALPHA_START} → {SARSA_ALPHA_END} (adaptativo)")
    print(f"  - Epsilon decay: {SARSA_EPSILON_DECAY}, min: {SARSA_EPSILON_MIN}")
    print(f"  - Episódios: {SARSA_EPISODES}")
    print(f"  - UCB Exploration: {'Habilitado' if SARSA_USE_UCB else 'Desabilitado'}")
    
    # ========================================================================
    # TREINAMENTO Q-LEARNING
    # ========================================================================
    print("\n" + "=" * 70)
    print("TREINANDO Q-LEARNING (Off-policy)")
    print("=" * 70)
    
    qlearning_agent = QLearningAgent(
        state_shape=STATE_SHAPE,
        n_actions=N_ACTIONS,
        alpha=QL_ALPHA,
        gamma=GAMMA,
        epsilon=EPSILON,
        epsilon_decay=QL_EPSILON_DECAY,
        epsilon_min=QL_EPSILON_MIN
    )
    
    qlearning_rewards = train_agent(
        agent=qlearning_agent,
        bins=bins,
        episodes=QL_EPISODES,
        is_sarsa=False,
        use_reward_shaping=True,
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
    print("TREINANDO SARSA (On-policy com melhorias)")
    print("=" * 70)
    
    # Reset da seed para comparação justa
    np.random.seed(42)
    
    sarsa_agent = SarsaAgent(
        state_shape=STATE_SHAPE,
        n_actions=N_ACTIONS,
        alpha=SARSA_ALPHA_START,  # Será adaptado durante treino
        gamma=GAMMA,
        epsilon=EPSILON,
        epsilon_decay=SARSA_EPSILON_DECAY,
        epsilon_min=SARSA_EPSILON_MIN
    )
    
    # INICIALIZAÇÃO OTIMISTA: força exploração eficiente no início
    sarsa_agent.q_table = np.ones(sarsa_agent.q_table.shape) * 50.0
    print("✓ Q-table inicializada otimisticamente (50.0)")
    
    sarsa_rewards = train_agent(
        agent=sarsa_agent,
        bins=bins,
        episodes=SARSA_EPISODES,
        is_sarsa=True,
        use_reward_shaping=True,
        use_ucb=SARSA_USE_UCB,
        alpha_start=SARSA_ALPHA_START,
        alpha_end=SARSA_ALPHA_END,
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
    plt.figure(figsize=(14, 7))
    
    # Plot das curvas suavizadas (cada uma com seu próprio range)
    qlearning_range = range(window - 1, len(qlearning_rewards))
    sarsa_range = range(window - 1, len(sarsa_rewards))
    
    plt.plot(qlearning_range, qlearning_smooth, 
             label=f'Q-Learning ({len(qlearning_rewards)} eps, final: {qlearning_final:.1f} ts)', 
             linewidth=2, color='blue', alpha=0.8)
    plt.plot(sarsa_range, sarsa_smooth, 
             label=f'SARSA ({len(sarsa_rewards)} eps, final: {sarsa_final:.1f} ts)', 
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
