import gymnasium as gym
import pygame
import numpy as np

time_s = 10
theta_max_deg = 45
n_bins_x = 5
n_bins_theta = 5


last_action = 0
count = 0
total = 0


env = gym.make("CartPole-v1", render_mode="human")
env = env.unwrapped
env.theta_threshold_radians = theta_max_deg * 3.14159 / 180

obs, info = env.reset()

x_low = env.observation_space.low[0]
x_high = env.observation_space.high[0]
theta_low = env.observation_space.low[2]
theta_high = env.observation_space.high[2]


pygame.init()
pygame.display.set_caption("Bang-Bang Controller")
clock = pygame.time.Clock()
start_time = pygame.time.get_ticks()

# leitor de ações
def read_bangbang_action():
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        return 0
    elif keys[pygame.K_RIGHT]:
        return 1
    else:
        return None

while True:

    elapsed = pygame.time.get_ticks() - start_time
    
    # reseta a simulação depois de atingir o limite te tempo
    if elapsed >= time_s*1000:
        print(f"tempo máximo alcançado")
        print("frequencia no estado alvo:", (count/total)*(elapsed/(time_s*1000)))

        count = 0
        total = 0
        last_action = 0
        obs, info = env.reset()
        start_time = pygame.time.get_ticks()
        continue 

    # rncerrar a simulação ao fechar a janela
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            env.close()
            exit()

    # controle bang-bang
    action = read_bangbang_action()
    if action is None:
        action = last_action
    else:
        last_action = action

    obs, reward, terminated, truncated, info = env.step(action)
    x, x_dot, theta, theta_dot = obs


    total += 1
    #criação do espaço de estados discreto
    x_discrete = np.digitize(x, np.linspace(x_low, x_high, n_bins_x - 1))
    theta_discrete = np.digitize(theta, np.linspace(theta_low, theta_high, n_bins_theta - 1))

    # verificar se o estado atual é o estado desejado
    if  x_discrete == (n_bins_x - 1)//2 and theta_discrete == (n_bins_theta - 1)//2:
        count += 1

    #print('estado atual:', x_discrete, theta_discrete,'estado_alvo =', (n_bins_x - 1)//2, (n_bins_theta - 1)//2)

    # reseta o ambiente ao antingir o angulo maximo ou uma parede
    if terminated or truncated:
        print(f"limite de theta ou x atingido")
        print("frequencia no estado alvo:", (count/total)*(elapsed/(time_s*1000)))
        count = 0
        total = 0
        last_action = 0
        obs, info = env.reset()
        start_time = pygame.time.get_ticks()  # reseta timer

    clock.tick(13)
    

