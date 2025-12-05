"""
Módulo de ambiente para CartPole com discretização do espaço de estados.

Este módulo fornece funções para criar bins de discretização e converter
estados contínuos em índices discretos, permitindo uso de algoritmos tabulares
de Reinforcement Learning.
"""

import numpy as np
import gymnasium as gym


def create_bins():
    """
    Cria os bins para discretização do espaço de estados do CartPole.
    
    O espaço de estados contínuo (4D) é discretizado em:
    - Cart Position (x): 6 bins
    - Cart Velocity (x_dot): 6 bins (limites fixos: -0.5 a 0.5 m/s)
    - Pole Angle (theta): 10 bins
    - Pole Angular Velocity (theta_dot): 10 bins (limites fixos: -50 a 50 deg/s)
    
    Returns:
        tuple: (cart_pos_bins, cart_vel_bins, pole_angle_bins, pole_ang_vel_bins)
               Cada elemento é um array numpy com os limites dos bins.
    """
    # Cart Position: usa limites do ambiente (aproximadamente -2.4 a 2.4)
    cart_pos_bins = np.linspace(-2.4, 2.4, 6 - 1)
    
    # Cart Velocity: limites fixos para capturar movimento sutil
    cart_vel_bins = np.linspace(-0.5, 0.5, 6 - 1)
    
    # Pole Angle: usa limites do ambiente (aproximadamente -0.209 a 0.209 rad = ±12°)
    # Mas vamos usar um range maior para maior robustez
    pole_angle_bins = np.linspace(-0.418, 0.418, 10 - 1)  # ±24°
    
    # Pole Angular Velocity: limites fixos convertidos para rad/s
    # 50 deg/s = 50 * pi/180 ≈ 0.873 rad/s
    pole_ang_vel_bins = np.linspace(-0.873, 0.873, 10 - 1)
    
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
