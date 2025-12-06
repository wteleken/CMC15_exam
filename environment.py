"""
Módulo de ambiente para CartPole com discretização do espaço de estados.

Este módulo fornece funções para criar bins de discretização e converter
estados contínuos em índices discretos, permitindo uso de algoritmos tabulares
de Reinforcement Learning.
"""

import numpy as np
import gymnasium as gym


def adaptive_linspace(start, end, num, critical_center=0.0, concentration=2.0):
    """
    Cria bins não-uniformes com maior resolução próximo ao ponto crítico.
    
    FASE 2: Binning adaptativo para concentrar estados nas regiões críticas.
    
    Args:
        start (float): Valor inicial do range
        end (float): Valor final do range
        num (int): Número de bins
        critical_center (float): Ponto onde concentrar bins (padrão: 0.0)
        concentration (float): Fator de concentração (>1 = mais concentrado)
    
    Returns:
        np.ndarray: Array de bins não-uniformes
    """
    # Cria espaçamento uniforme em [-1, 1]
    uniform = np.linspace(-1, 1, num)
    
    # Aplica função não-linear para concentrar no centro
    # sign(x) * |x|^concentration redistribui pontos
    concentrated = np.sign(uniform) * np.abs(uniform) ** concentration
    
    # Escala para o range desejado, centrado no ponto crítico
    half_range = (end - start) / 2
    bins = critical_center + concentrated * half_range
    
    return bins


def create_bins(state_shape=None):
    """
    Cria os bins para discretização do espaço de estados do CartPole.
    
    O espaço de estados contínuo (4D) é discretizado em:
    - Cart Position (x): 6 bins (menor resolução - menos crítico)
    - Cart Velocity (x_dot): 6 bins (menor resolução - menos crítico)
    - Pole Angle (theta): 18 bins (alta resolução - CRÍTICO)
    - Pole Angular Velocity (theta_dot): 18 bins (alta resolução - CRÍTICO)
    
    Novo shape padrão: (6, 6, 18, 18) = 11,664 estados (vs. 9,216 anterior)
    Concentra resolução nas variáveis mais críticas para estabilidade.
    
    Returns:
        tuple: (cart_pos_bins, cart_vel_bins, pole_angle_bins, pole_ang_vel_bins)
               Cada elemento é um array numpy com os limites dos bins.
    
    Args:
        state_shape (tuple): Formato (n_cart_pos, n_cart_vel, n_pole_angle, n_pole_vel).
                            Se None, usa padrão (6, 6, 18, 18).
    """
    # Padrão refinado: 6x6x18x18 = 11,664 estados
    # Menor resolução em posição/velocidade do carrinho
    # Maior resolução em ângulo/velocidade angular (variáveis críticas)
    if state_shape is None:
        state_shape = (6, 6, 18, 18)
    
    n_cart_pos, n_cart_vel, n_pole_angle, n_pole_vel = state_shape
    
    # FASE 2: Binning adaptativo não-uniforme
    # Concentra resolução nas regiões críticas (θ≈0, velocidades baixas)
    
    # Bins para posição e velocidade do carrinho
    cart_pos_bins = adaptive_linspace(-2.4, 2.4, n_cart_pos - 1, critical_center=0.0, concentration=1.5)
    cart_vel_bins = adaptive_linspace(-1.5, 1.5, n_cart_vel - 1, critical_center=0.0, concentration=2.0)
    
    # Bins para ângulo e velocidade angular (mais críticos)
    pole_angle_bins = adaptive_linspace(-0.418, 0.418, n_pole_angle - 1, critical_center=0.0, concentration=2.5)
    pole_ang_vel_bins = adaptive_linspace(-2.618, 2.618, n_pole_vel - 1, critical_center=0.0, concentration=2.0)
    
    return (cart_pos_bins, cart_vel_bins, pole_angle_bins, pole_ang_vel_bins)


def discretize_state(obs, bins):
    """
    Converte um estado contínuo em índices discretos.
    
    Args:
        obs (array-like): Observação do ambiente [x, x_dot, theta, theta_dot]
        bins (tuple): Tupla com 4 arrays de bins (retorno de create_bins())
    
    Returns:
        tuple: (x_idx, x_dot_idx, theta_idx, theta_dot_idx) - índices discretos
    """
    cart_pos_bins, cart_vel_bins, pole_angle_bins, pole_ang_vel_bins = bins
    
    x, x_dot, theta, theta_dot = obs
    
    # np.digitize retorna índice do bin (0 a n_bins)
    x_idx = np.digitize(x, cart_pos_bins)
    x_dot_idx = np.digitize(x_dot, cart_vel_bins)
    theta_idx = np.digitize(theta, pole_angle_bins)
    theta_dot_idx = np.digitize(theta_dot, pole_ang_vel_bins)
    
    return (x_idx, x_dot_idx, theta_idx, theta_dot_idx)


def create_environment(render_mode=None, theta_threshold_deg=45):
    """
    Cria e configura o ambiente CartPole-v1.
    
    Args:
        render_mode (str, optional): Modo de renderização ('human', 'rgb_array', None)
        theta_threshold_deg (float): Ângulo máximo permitido em graus (padrão: 45°)
    
    Returns:
        gym.Env: Ambiente CartPole configurado
    """
    env = gym.make("CartPole-v1", render_mode=render_mode)
    env = env.unwrapped
    
    # Configura threshold do ângulo (conversão deg -> rad)
    env.theta_threshold_radians = theta_threshold_deg * np.pi / 180
    
    return env
