## Análise do Controle do Pêndulo Invertido


### Integrantes do grupo
Daniel da Silveira Sahadi

Pablo Carvalho do Nascimentos dos Santos

Thiago Galante Pereira

Willian Nelton Teleken


### a) Dificuldade de Controle Humano

A dificuldade de controlar o sistema como humano depende de alguns fatores principais:

#### **Velocidade do Sistema**
- a velocidade do sistema pode ser determinada pelo numero de tiques do clock, com um numero de ticks menor podemos enviar mais comandos ao sistema e tambem aumentamos o tempo de reação e processamento do jogador humano para a situação atual

#### **Complexidade do Objetivo**
- **Objetivo simples**: Maximizar tempo no ambiente: bastante factível
- **Objetivo complexo**: Manter a posição (x = 0, θ = 0): consideravelmente difícil

- Se o objetivo for controlar o carrinho para mante-lo na posição x = 0 com θ = 0 o o problema se torna muito dificil para um humano, apesar de conseguirmos utilizar noções de fisica para maximizar o tempo do carrinho no ambiente, mante-lo estavel na posição x = 0 e θ = 0 é consideravelmente mais dificil

### b) Como o progresso no aprendizado do controle manual poderia ser estimado?

#### **Problema com Tempo**

- Tentando controlar o carrinho na posição x = 0 e θ = 0 logo notamos que tempo não é uma boa metrica para mensurar essa capacidade. uma vez que conseguimos deixar o carrinho por muito tempo no ambiente e ainda bastante longe desse estado, precisamos de uma metrica melhor

#### **Solução possível: Discretização do Espaço**
- observando o conjunto de valores que o sistema pode assumir notamos que x e θ são limitados no problema, dessa forma podemos efetuar uma discretização dessas variaveis para o espaço de estados e podemos mensurar a performance de aprendizado do ser humano como a capacidade de manter o pendulo no estado mais estavel, considerando como perfomance a frequencia no estado mais estavel. Todavia, isso pode levar a uma politica de tentar encerrar o jogo o mais rapido possivel, pois, ao acabar o jogo rapidamente, o estado inicial (x = 0 e θ = 0) representa uma grande parcela de todos os estados. Assim uma solução multiplicar essa frequencia no estado desejado pelo percentual do tempo de simulação. Com isso, simulações curtas com uma frequencia alta no estado zero são fortemente penalizadas.

---

## c) Implementação de Algoritmos de Aprendizado por Reforço

### Objetivo
Implementar e comparar dois algoritmos tabulares de Reinforcement Learning — **Q-Learning** (off-policy) e **SARSA** (on-policy) — para controle autônomo do pêndulo invertido no ambiente CartPole-v1.

### Fundamentação Teórica

#### Q-Learning (Off-policy)
O Q-Learning é um algoritmo de aprendizado por diferença temporal que aprende a política ótima independentemente da política seguida durante o treinamento. A regra de atualização é:

$$Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]$$

Onde:
- $Q(s,a)$ é o valor da ação $a$ no estado $s$
- $\alpha$ é a taxa de aprendizado
- $r$ é a recompensa recebida
- $\gamma$ é o fator de desconto
- $s'$ é o próximo estado
- $\max_{a'} Q(s',a')$ é o valor da melhor ação possível no próximo estado

**Característica chave**: Usa a melhor ação futura possível (max), independente da ação que será realmente tomada.

#### SARSA (On-policy)
O SARSA (State-Action-Reward-State-Action) aprende o valor da política que está sendo seguida. A regra de atualização é:

$$Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma Q(s',a') - Q(s,a) \right]$$

Onde $a'$ é a próxima ação **realmente escolhida** pela política epsilon-greedy.

**Característica chave**: Usa a próxima ação real que será executada, tornando-o mais conservador em ambientes com risco.

**Referência**: Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

### Arquitetura de Implementação

O projeto foi estruturado de forma modular para facilitar manutenção e experimentação:

```
CMC15_exam/
├── manual_control.py       # Controle humano (entrega parcial - partes a e b)
├── environment.py          # Discretização do espaço de estados
├── agents.py               # Implementação dos algoritmos RL
├── train.py                # Lógica de treinamento genérica
├── comparison.py           # Script principal de comparação
└── README.md              # Este arquivo
```

#### `environment.py`
- **Função `create_bins()`**: Cria bins para discretização 4D do espaço de estados
  - Cart Position (x): 6 bins
  - Cart Velocity (ẋ): 6 bins (limites: -0.5 a 0.5 m/s)
  - Pole Angle (θ): 10 bins  
  - Pole Angular Velocity (θ̇): 10 bins (limites: -50 a 50 deg/s)
  - **Total**: 6 × 6 × 10 × 10 = 3.600 estados discretos

- **Função `discretize_state()`**: Converte observações contínuas em índices discretos usando `np.digitize`

#### `agents.py`
- **`BaseAgent`**: Classe base abstrata com Q-table (inicializada com zeros) e política epsilon-greedy
- **`QLearningAgent`**: Implementa atualização off-policy usando max(Q(s',a'))
- **`SarsaAgent`**: Implementa atualização on-policy usando Q(s',a') real

#### `train.py`
- **`train_agent()`**: Função genérica que treina qualquer agente
  - Suporta tanto Q-Learning quanto SARSA (flag `is_sarsa`)
  - Treina sem renderização gráfica para maior velocidade
  - Retorna histórico de recompensas por episódio

#### `comparison.py`
- Script principal que executa toda a pipeline de comparação
- Hiperparâmetros fixos: α=0.1, γ=0.99, ε₀=1.0 (decay=0.995, min=0.01)
- Treina ambos os algoritmos por 5.000 episódios
- Gera gráfico comparativo com médias móveis (janela=100 episódios)
- Salva Q-tables treinadas para análise posterior

### Execução

#### Instalar dependências
```bash
pip install -r requirements.txt
```

#### Executar comparação
```bash
python comparison.py
```

Isso irá:
1. Treinar o agente Q-Learning (≈2-5 minutos)
2. Treinar o agente SARSA (≈2-5 minutos)
3. Gerar gráfico comparativo (`comparison_result.png`)
4. Salvar Q-tables treinadas (`.pkl`)

#### Testar controle manual (entrega parcial)
```bash
python manual_control.py
```
Use as teclas ← e → para controlar o carrinho.

### Resultados Esperados

#### Convergência
Ambos os algoritmos devem convergir para políticas efetivas, atingindo recompensas médias próximas ao máximo do ambiente (500 timesteps no CartPole-v1).

#### Diferenças Observadas
- **Q-Learning**: Tipicamente aprende mais rápido, mas pode ser menos estável durante o treinamento
- **SARSA**: Tende a ser mais conservador e estável, pois considera a exploração na atualização

#### Interpretação
A comparação demonstra empiricamente a diferença entre aprendizado **on-policy** (SARSA aprende sobre a política que executa) e **off-policy** (Q-Learning aprende a política ótima independentemente da exploração).

### Arquivos Gerados
- `comparison_result.png`: Gráfico comparativo das curvas de aprendizado
- `qlearning_qtable.pkl`: Q-table treinada do Q-Learning
- `sarsa_qtable.pkl`: Q-table treinada do SARSA

### Extensões Futuras
- Implementar visualização das políticas aprendidas
- Testar diferentes configurações de bins e hiperparâmetros
- Comparar com algoritmos baseados em função (DQN, Policy Gradient)
- Analisar distribuição de estados visitados