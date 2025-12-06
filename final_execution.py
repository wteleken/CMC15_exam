"""
Script Final de Treinamento e Visualização para Projeto CMC-15.

Este script executa o treinamento completo dos 3 algoritmos e gera todos os
gráficos necessários para o relatório final.

Autores: Daniel Sahadi, Pablo Carvalho, Thiago Galante, Willian Teleken
Data: Dezembro 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime

# Imports do projeto
from agents import QLearningAgent, SarsaAgent, SarsaLambdaAgent
from environment import create_bins
from train import train_agent


def moving_average(data, window=100):
    """
    Calcula média móvel para suavização de curvas.
    
    Args:
        data (list): Dados originais
        window (int): Tamanho da janela de média
    
    Returns:
        np.ndarray: Dados suavizados
    """
    if len(data) < window:
        return np.array(data)
    
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / window


def train_qlearning(episodes=25000, bins=None, verbose=True):
    """
    Treina agente Q-Learning com configuração padrão.
    
    Args:
        episodes (int): Número de episódios
        bins (tuple): Bins de discretização (se None, usa padrão 8,8,12,12)
        verbose (bool): Imprimir progresso
    
    Returns:
        tuple: (agent, episode_rewards, bins)
    """
    if verbose:
        print("\n" + "="*60)
        print("TREINANDO Q-LEARNING")
        print("="*60)
        print(f"Episódios: {episodes}")
        print(f"Alpha: 0.15 (fixo)")
        print(f"Gamma: 0.99")
        print(f"Epsilon: 1.0 → 0.01 (decay=0.995)")
        print(f"Discretização: (8, 8, 12, 12)")
    
    # Bins padrão antigo (8,8,12,12) para Q-Learning
    if bins is None:
        # Força discretização antiga
        from environment import adaptive_linspace
        import numpy as np
        cart_pos_bins = adaptive_linspace(-2.4, 2.4, 7, critical_center=0.0, concentration=1.5)
        cart_vel_bins = adaptive_linspace(-1.5, 1.5, 7, critical_center=0.0, concentration=2.0)
        pole_angle_bins = adaptive_linspace(-0.418, 0.418, 11, critical_center=0.0, concentration=2.5)
        pole_ang_vel_bins = adaptive_linspace(-2.0, 2.0, 11, critical_center=0.0, concentration=2.0)
        bins = (cart_pos_bins, cart_vel_bins, pole_angle_bins, pole_ang_vel_bins)
    
    state_shape = tuple(len(b) + 1 for b in bins)
    
    # Inicializa agente
    agent = QLearningAgent(
        state_shape=state_shape,
        n_actions=2,
        alpha=0.15,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    # Treina
    rewards = train_agent(
        agent=agent,
        bins=bins,
        episodes=episodes,
        is_sarsa=False,
        use_reward_shaping=False,
        use_ucb=False,
        alpha_start=None,
        alpha_end=None,
        verbose=verbose
    )
    
    if verbose:
        mean_reward = np.mean(rewards[-100:])
        print(f"\n✅ Q-Learning treinado!")
        print(f"Performance final (últimos 100 eps): {mean_reward:.1f} ± {np.std(rewards[-100:]):.1f} ts")
    
    return agent, rewards, bins


def train_sarsa_optimized(episodes=50000, bins=None, verbose=True):
    """
    Treina agente SARSA Otimizado com UCB e alpha adaptativo.
    
    Args:
        episodes (int): Número de episódios
        bins (tuple): Bins de discretização (se None, usa padrão 8,8,12,12)
        verbose (bool): Imprimir progresso
    
    Returns:
        tuple: (agent, episode_rewards, bins)
    """
    if verbose:
        print("\n" + "="*60)
        print("TREINANDO SARSA OTIMIZADO")
        print("="*60)
        print(f"Episódios: {episodes}")
        print(f"Alpha: 0.3 → 0.05 (adaptativo)")
        print(f"Gamma: 0.99")
        print(f"UCB Exploration: True (primeiros 30%)")
        print(f"Epsilon: 1.0 → 0.01 (decay=0.995)")
        print(f"Discretização: (8, 8, 12, 12)")
    
    # Bins padrão antigo (8,8,12,12) para SARSA
    if bins is None:
        # Força discretização antiga
        from environment import adaptive_linspace
        import numpy as np
        cart_pos_bins = adaptive_linspace(-2.4, 2.4, 7, critical_center=0.0, concentration=1.5)
        cart_vel_bins = adaptive_linspace(-1.5, 1.5, 7, critical_center=0.0, concentration=2.0)
        pole_angle_bins = adaptive_linspace(-0.418, 0.418, 11, critical_center=0.0, concentration=2.5)
        pole_ang_vel_bins = adaptive_linspace(-2.0, 2.0, 11, critical_center=0.0, concentration=2.0)
        bins = (cart_pos_bins, cart_vel_bins, pole_angle_bins, pole_ang_vel_bins)
    
    state_shape = tuple(len(b) + 1 for b in bins)
    
    # Inicializa agente
    agent = SarsaAgent(
        state_shape=state_shape,
        n_actions=2,
        alpha=0.3,  # Será adaptado durante treino
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    # Treina com UCB e alpha adaptativo
    rewards = train_agent(
        agent=agent,
        bins=bins,
        episodes=episodes,
        is_sarsa=True,
        use_reward_shaping=False,
        use_ucb=True,  # UCB nos primeiros 30%
        alpha_start=0.3,
        alpha_end=0.05,
        verbose=verbose
    )
    
    if verbose:
        mean_reward = np.mean(rewards[-100:])
        print(f"\n✅ SARSA Otimizado treinado!")
        print(f"Performance final (últimos 100 eps): {mean_reward:.1f} ± {np.std(rewards[-100:]):.1f} ts")
    
    return agent, rewards, bins


def train_sarsa_lambda(episodes=5000, verbose=True):
    """
    Treina agente SARSA(λ) com Eligibility Traces.
    
    Args:
        episodes (int): Número de episódios
        verbose (bool): Imprimir progresso
    
    Returns:
        tuple: (agent, episode_rewards, bins)
    """
    if verbose:
        print("\n" + "="*60)
        print("TREINANDO SARSA(λ) COM ELIGIBILITY TRACES")
        print("="*60)
        print(f"Episódios: {episodes}")
        print(f"Alpha: 0.15 (fixo)")
        print(f"Gamma: 0.99")
        print(f"Lambda: 0.9")
        print(f"Bins refinados: (6, 6, 18, 18)")
        print(f"Epsilon: 1.0 → 0.01 (decay=0.9995)")
    
    # Bins refinados (6,6,18,18) - usa o padrão atual do environment.py
    bins = create_bins()  # Usa padrão (6,6,18,18)
    state_shape = tuple(len(b) + 1 for b in bins)
    
    # Inicializa agente
    agent = SarsaLambdaAgent(
        state_shape=state_shape,
        n_actions=2,
        alpha=0.15,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.9995,  # Decay mais lento para SARSA(λ)
        epsilon_min=0.01,
        lambda_factor=0.9,
        trace_threshold=0.001
    )
    
    # Treina (train_agent detecta automaticamente reset_traces())
    rewards = train_agent(
        agent=agent,
        bins=bins,
        episodes=episodes,
        is_sarsa=True,  # SARSA(λ) é on-policy como SARSA
        use_reward_shaping=False,
        use_ucb=False,
        alpha_start=None,
        alpha_end=None,
        verbose=verbose
    )
    
    if verbose:
        mean_reward = np.mean(rewards[-100:])
        print(f"\n✅ SARSA(λ) treinado!")
        print(f"Performance final (últimos 100 eps): {mean_reward:.1f} ± {np.std(rewards[-100:]):.1f} ts")
    
    return agent, rewards, bins


def plot_comparison(results_dict, save_path='comparison_result.png'):
    """
    Gera Gráfico 1: Comparação de Performance entre os 3 algoritmos.
    
    Args:
        results_dict (dict): Dicionário com formato:
            {
                'Q-Learning': {'rewards': [...], 'color': 'blue'},
                'SARSA Otimizado': {'rewards': [...], 'color': 'green'},
                'SARSA(λ)': {'rewards': [...], 'color': 'red'}
            }
        save_path (str): Caminho para salvar o gráfico
    """
    print("\n" + "="*60)
    print("GERANDO GRÁFICO 1: COMPARAÇÃO DE PERFORMANCE")
    print("="*60)
    
    plt.figure(figsize=(14, 8))
    
    # Eixo X até 50k episódios
    max_episodes = 50000
    
    for label, data in results_dict.items():
        rewards = data['rewards']
        color = data['color']
        
        # Média móvel (window=100)
        smoothed = moving_average(rewards, window=100)
        
        # Eixo X ajustado
        x_values = np.arange(len(smoothed))
        
        # Média final (últimos 100 episódios)
        final_mean = np.mean(rewards[-100:])
        
        # Plota curva
        plt.plot(
            x_values,
            smoothed,
            label=f"{label} (μ={final_mean:.1f} ts)",
            color=color,
            linewidth=2,
            alpha=0.8
        )
    
    # Configurações do gráfico
    plt.xlabel('Episódios', fontsize=14, fontweight='bold')
    plt.ylabel('Recompensa (Timesteps)', fontsize=14, fontweight='bold')
    plt.title('Comparação de Performance - Reinforcement Learning', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlim(0, max_episodes)
    plt.ylim(0, None)  # Auto-ajusta limite superior
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=12, loc='lower right', framealpha=0.9)
    plt.tight_layout()
    
    # Salva
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Gráfico salvo em: {save_path}")
    plt.close()


def plot_policy_heatmaps(agents_dict, bins_dict, save_path='policy_heatmaps.png'):
    """
    Gera Gráfico 2: Heatmaps da Política (3 subplots).
    
    Para cada agente, fixa x e x_dot no centro e plota um heatmap 2D
    mostrando a ação escolhida (argmax Q-table) para cada combinação
    de theta e theta_dot.
    
    Args:
        agents_dict (dict): Dicionário com agentes treinados
            {
                'Q-Learning': agent1,
                'SARSA Otimizado': agent2,
                'SARSA(λ)': agent3
            }
        bins_dict (dict): Dicionário com bins usados por cada agente
        save_path (str): Caminho para salvar o gráfico
    """
    print("\n" + "="*60)
    print("GERANDO GRÁFICO 2: HEATMAPS DA POLÍTICA")
    print("="*60)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (label, agent) in enumerate(agents_dict.items()):
        ax = axes[idx]
        bins = bins_dict[label]
        
        # Dimensões da Q-table
        state_shape = agent.state_shape
        n_x, n_x_dot, n_theta, n_theta_dot = state_shape
        
        # Fixa x e x_dot no índice central
        x_center = n_x // 2
        x_dot_center = n_x_dot // 2
        
        # Cria matriz para o heatmap: theta (X) vs theta_dot (Y)
        policy_map = np.zeros((n_theta_dot, n_theta))
        
        for theta_idx in range(n_theta):
            for theta_dot_idx in range(n_theta_dot):
                # Estado: (x_center, x_dot_center, theta_idx, theta_dot_idx)
                state = (x_center, x_dot_center, theta_idx, theta_dot_idx)
                
                # Ação escolhida = argmax Q(s, a)
                action = np.argmax(agent.q_table[state])
                policy_map[theta_dot_idx, theta_idx] = action
        
        # Plota heatmap
        im = ax.imshow(
            policy_map,
            cmap='RdYlGn',  # Red (0=Left), Green (1=Right)
            aspect='auto',
            origin='lower',
            interpolation='nearest'
        )
        
        # Configurações do subplot
        ax.set_title(f'{label}\n(x={x_center}, ẋ={x_dot_center})', 
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Índice θ (Ângulo)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Índice θ̇ (Vel. Angular)', fontsize=12, fontweight='bold')
        
        # Adiciona colorbar
        cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
        cbar.set_label('Ação', fontsize=11, fontweight='bold')
        cbar.ax.set_yticklabels(['← Esquerda (0)', 'Direita → (1)'])
        
        # Grid
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    
    # Título geral
    fig.suptitle('Fronteira de Decisão das Políticas Aprendidas', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Salva
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Gráfico salvo em: {save_path}")
    plt.close()


def save_models(agents_dict, prefix='final'):
    """
    Salva as Q-tables dos agentes treinados.
    
    Args:
        agents_dict (dict): Dicionário com agentes
        prefix (str): Prefixo para os arquivos
    """
    print("\n" + "="*60)
    print("SALVANDO MODELOS")
    print("="*60)
    
    for label, agent in agents_dict.items():
        # Normaliza nome do arquivo
        filename = f"{prefix}_{label.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('λ', 'lambda')}_qtable.pkl"
        
        with open(filename, 'wb') as f:
            pickle.dump(agent.q_table, f)
        
        print(f"✅ {label}: {filename}")


def main():
    """
    Execução principal do script.
    
    1. Treina os 3 modelos
    2. Gera os 2 gráficos
    3. Salva as Q-tables
    """
    print("\n" + "╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "  SCRIPT FINAL DE TREINAMENTO E VISUALIZAÇÃO".center(58) + "║")
    print("║" + "  Projeto CMC-15 - Reinforcement Learning".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")
    print(f"\nData: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    
    # ========================================================================
    # FASE 1: TREINAMENTO DOS 3 MODELOS
    # ========================================================================
    
    # 1.1. Q-Learning (25k eps, bins padrão)
    qlearning_agent, qlearning_rewards, qlearning_bins = train_qlearning(
        episodes=25000,
        bins=None,  # Usa padrão (8,8,12,12)
        verbose=True
    )
    
    # 1.2. SARSA Otimizado (50k eps, bins padrão, UCB, alpha adaptativo)
    sarsa_agent, sarsa_rewards, sarsa_bins = train_sarsa_optimized(
        episodes=50000,
        bins=None,  # Usa padrão (8,8,12,12)
        verbose=True
    )
    
    # 1.3. SARSA(λ) (5k eps, bins refinados)
    sarsa_lambda_agent, sarsa_lambda_rewards, sarsa_lambda_bins = train_sarsa_lambda(
        episodes=5000,
        verbose=True
    )
    
    # ========================================================================
    # FASE 2: GRÁFICO 1 - COMPARAÇÃO DE PERFORMANCE
    # ========================================================================
    
    results_dict = {
        'Q-Learning': {
            'rewards': qlearning_rewards,
            'color': 'blue'
        },
        'SARSA Otimizado': {
            'rewards': sarsa_rewards,
            'color': 'green'
        },
        'SARSA(λ)': {
            'rewards': sarsa_lambda_rewards,
            'color': 'red'
        }
    }
    
    plot_comparison(results_dict, save_path='comparison_result.png')
    
    # ========================================================================
    # FASE 3: GRÁFICO 2 - HEATMAPS DA POLÍTICA
    # ========================================================================
    
    agents_dict = {
        'Q-Learning': qlearning_agent,
        'SARSA Otimizado': sarsa_agent,
        'SARSA(λ)': sarsa_lambda_agent
    }
    
    bins_dict = {
        'Q-Learning': qlearning_bins,
        'SARSA Otimizado': sarsa_bins,
        'SARSA(λ)': sarsa_lambda_bins
    }
    
    plot_policy_heatmaps(agents_dict, bins_dict, save_path='policy_heatmaps.png')
    
    # ========================================================================
    # FASE 4: SALVAR MODELOS
    # ========================================================================
    
    save_models(agents_dict, prefix='final')
    
    # ========================================================================
    # RESUMO FINAL
    # ========================================================================
    
    print("\n" + "╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "  ✅ EXECUÇÃO CONCLUÍDA COM SUCESSO!".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")
    
    print("\n📊 ARQUIVOS GERADOS:")
    print("  ├─ comparison_result.png       (Gráfico 1: Comparação)")
    print("  ├─ policy_heatmaps.png         (Gráfico 2: Heatmaps)")
    print("  ├─ final_q-learning_qtable.pkl")
    print("  ├─ final_sarsa_otimizado_qtable.pkl")
    print("  └─ final_sarsa_lambda_qtable.pkl")
    
    print("\n🏆 RESULTADOS FINAIS:")
    for label, agent in agents_dict.items():
        if label == 'Q-Learning':
            rewards = qlearning_rewards
        elif label == 'SARSA Otimizado':
            rewards = sarsa_rewards
        else:
            rewards = sarsa_lambda_rewards
        
        mean = np.mean(rewards[-100:])
        std = np.std(rewards[-100:])
        print(f"  • {label:20s}: {mean:6.1f} ± {std:5.1f} ts")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
