# Notas de Implementação - Projeto CartPole RL

> **Objetivo**: Este documento facilita a compreensão do código e a escrita do relatório final. Contém decisões de design, justificativas técnicas e sugestões de análise.

---

## 📋 Visão Geral do Projeto

Este projeto implementa e compara dois algoritmos clássicos de Reinforcement Learning (RL) tabular:
- **Q-Learning** (off-policy)
- **SARSA** (on-policy)

**Ambiente**: CartPole-v1 do Gymnasium (Pêndulo Invertido)  
**Objetivo**: Aprender política para manter o pêndulo equilibrado o máximo de tempo possível

---

## 🏗️ Arquitetura Modular

### Por que separamos em módulos?

1. **Reutilização**: Mesma discretização e lógica de treinamento para ambos algoritmos
2. **Manutenção**: Mudanças em um componente não afetam outros
3. **Testabilidade**: Cada módulo pode ser testado independentemente
4. **Clareza**: Código organizado é mais fácil de entender e explicar no relatório

### Estrutura de Arquivos

```
CMC15_exam/
│
├── manual_control.py       ← Código original (partes a e b) - PRESERVADO
│
├── environment.py          ← Discretização do espaço de estados
│   ├── create_bins()       → Cria bins para discretização 4D
│   ├── discretize_state()  → Converte estado contínuo → índices discretos
│   └── create_environment()→ Configura CartPole-v1
│
├── agents.py               ← Implementação dos algoritmos RL
│   ├── BaseAgent           → Classe base (Q-table, epsilon-greedy)
│   ├── QLearningAgent      → Atualização off-policy
│   └── SarsaAgent          → Atualização on-policy
│
├── train.py                ← Lógica de treinamento
│   └── train_agent()       → Loop genérico de episódios
│
├── comparison.py           ← Script principal (execução completa)
│   └── main()              → Treina ambos + gera gráfico
│
└── README.md               ← Documentação para o professor
```

---

## 🎯 Decisões de Design Importantes

### 1. Discretização do Espaço de Estados

**Por que discretizar?**
- CartPole tem espaço de estados **contínuo** (4 variáveis reais)
- Algoritmos tabulares (Q-Learning, SARSA) precisam de estados **discretos**
- Alternativa seria usar Deep RL (DQN), mas é mais complexo

**Configuração escolhida:**
```
Cart Position (x):         6 bins  → range padrão do ambiente
Cart Velocity (ẋ):         6 bins  → limites [-0.5, 0.5] m/s
Pole Angle (θ):           10 bins  → limites [-24°, 24°]
Pole Angular Velocity (θ̇): 10 bins  → limites [-50, 50] deg/s
────────────────────────────────────────────────────────
TOTAL: 6 × 6 × 10 × 10 = 3.600 estados discretos
```

**Justificativa para os limites fixos:**
- **Velocidades do carrinho**: Valores típicos ficam em [-0.5, 0.5] m/s. Limites fixos evitam bins extremos pouco visitados.
- **Velocidades angulares**: Convertido de graus/s para rad/s. Valores típicos concentrados nessa faixa.

**Trade-off:**
- Mais bins = maior precisão, mas Q-table maior e mais tempo para treinar
- Menos bins = treino mais rápido, mas perda de informação
- 3.600 estados é um bom meio-termo (gerenciável e informativo)

### 2. Hiperparâmetros

```python
ALPHA = 0.1           # Taxa de aprendizado
GAMMA = 0.99          # Fator de desconto
EPSILON = 1.0         # Exploração inicial (100%)
EPSILON_DECAY = 0.995 # Decaimento por episódio
EPSILON_MIN = 0.01    # Exploração mínima (1%)
EPISODES = 5000       # Número de episódios
```

**Justificativas:**
- **α = 0.1**: Taxa moderada. Valores muito altos causam instabilidade; muito baixos, lentidão.
- **γ = 0.99**: Valoriza recompensas futuras (horizonte longo). Adequado para CartPole.
- **ε-greedy com decay**: Começa explorando muito (ε=1.0) e gradualmente vira greedy (ε→0.01).
- **5000 episódios**: Suficiente para convergência sem ser excessivo.

### 3. Diferença Q-Learning vs SARSA no Código

#### Q-Learning (Off-policy)
```python
# Atualiza usando a MELHOR ação possível
max_next_q = np.max(self.q_table[next_state])
new_q = current_q + alpha * (reward + gamma * max_next_q - current_q)
```

**Loop de treino:**
```python
while not done:
    obs, reward, done, ... = env.step(action)
    next_state = discretize_state(obs)
    
    agent.update(state, action, reward, next_state)  # Não precisa de next_action
    
    state = next_state
    action = agent.select_action(state)  # Seleciona depois do update
```

#### SARSA (On-policy)
```python
# Atualiza usando a próxima ação REAL
next_q = self.q_table[next_state][next_action]  # next_action é parâmetro!
new_q = current_q + alpha * (reward + gamma * next_q - current_q)
```

**Loop de treino:**
```python
action = agent.select_action(state)  # Primeira ação

while not done:
    obs, reward, done, ... = env.step(action)
    next_state = discretize_state(obs)
    
    next_action = agent.select_action(next_state)  # Seleciona ANTES do update
    
    agent.update(state, action, reward, next_state, next_action)
    
    state = next_state
    action = next_action  # Usa a ação já escolhida
```

**Diferença fundamental:**
- Q-Learning: "Se eu fosse ótimo, qual seria minha melhor ação?" (otimista)
- SARSA: "Dada minha política atual (com exploração), o que vou fazer?" (conservador)

---

## 📊 Interpretação dos Resultados

### Métricas Importantes

1. **Curva de Aprendizado**: Recompensa média móvel ao longo dos episódios
   - Indica velocidade de convergência
   - Mostra estabilidade durante treinamento

2. **Recompensa Média Final** (últimos 100 episódios):
   - Indica performance final estabilizada
   - CartPole-v1 tem máximo de 500 timesteps

3. **Desvio Padrão**:
   - Indica consistência da política aprendida
   - Menor desvio = política mais estável

### Comportamentos Esperados

#### Q-Learning
- ✅ **Converge mais rápido** (aprende sobre política ótima desde o início)
- ⚠️ **Pode ser mais instável** (otimismo excessivo durante exploração)
- 📈 **Picos de recompensa mais cedo** no gráfico

#### SARSA
- ✅ **Mais estável durante treinamento** (conservador)
- ⚠️ **Pode convergir mais devagar** (aprende sobre política exploratória)
- 📈 **Curva mais suave**, menos oscilações

### Discussão para o Relatório

**Pergunta-chave**: Por que SARSA é chamado de "on-policy"?
- **Resposta**: Porque ele aprende o valor da política que está **executando** (incluindo exploração ε-greedy). Já o Q-Learning aprende a política ótima independente da exploração.

**Analogia útil**:
- **Q-Learning**: "Estudo como dirigir perfeitamente (sem erros) assistindo vídeos de pilotos profissionais"
- **SARSA**: "Aprendo a dirigir praticando, incluindo meus próprios erros e manobras exploratórias"

**Quando SARSA é melhor?**
- Ambientes com "penhascos" (cliff-walking)
- Quando exploração pode ser perigosa
- Quando queremos política segura

**Quando Q-Learning é melhor?**
- Ambientes onde exploração não tem custo alto
- Quando queremos convergência mais rápida
- CartPole é relativamente "seguro" para exploração

---

## 🔬 Análises Adicionais Sugeridas

### Para Enriquecer o Relatório

1. **Análise de Convergência**:
   - Em qual episódio cada algoritmo atinge 90% da performance máxima?
   - Quantos episódios até estabilização?

2. **Visualização da Política**:
   - Para alguns estados específicos, qual ação cada agente escolhe?
   - Exemplo: Estado (x=0, ẋ=0, θ=5°, θ̇=0) → Ação?

3. **Distribuição de Estados**:
   - Quais estados são mais visitados durante o treino?
   - Há estados nunca explorados?

4. **Sensibilidade de Hiperparâmetros**:
   - Como α = 0.05 vs α = 0.2 afeta convergência?
   - E se usarmos epsilon_decay = 0.99?

5. **Comparação com Controle Manual** (partes a e b):
   - Agente treinado vs humano: quem performa melhor?
   - Métrica da parte (b) aplicada aos agentes RL

### Código para Avaliação Visual (sugestão)

```python
# evaluate.py
import pickle
import numpy as np
from environment import create_environment, create_bins, discretize_state

# Carrega Q-table treinada
with open('qlearning_qtable.pkl', 'rb') as f:
    q_table = pickle.load(f)

# Cria ambiente com visualização
env = create_environment(render_mode='human')
bins = create_bins()

# Executa episódio com política greedy (epsilon=0)
obs, _ = env.reset()
done = False
total_reward = 0

while not done:
    state = discretize_state(obs, bins)
    action = np.argmax(q_table[state])  # Greedy
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    total_reward += reward

print(f"Recompensa total: {total_reward}")
env.close()
```

---

## 📝 Estrutura Sugerida para o Relatório

### 1. Introdução
- Contexto: Problema do pêndulo invertido
- Objetivo: Comparar Q-Learning e SARSA
- Justificativa: Demonstrar diferença on-policy vs off-policy

### 2. Fundamentação Teórica
- Aprendizado por Reforço (MDPs, Bellman)
- Métodos de Diferença Temporal (TD Learning)
- Q-Learning (fórmula + características)
- SARSA (fórmula + características)
- Comparação teórica (Tabela)

### 3. Metodologia
- Ambiente: CartPole-v1 (descrição, espaço de estados/ações)
- Discretização: Bins escolhidos e justificativa
- Hiperparâmetros: Valores e justificativa
- Arquitetura do código (módulos)
- Protocolo de treinamento (seeds, número de episódios)

### 4. Resultados
- Curvas de aprendizado (gráfico do comparison_result.png)
- Tabela comparativa (métricas finais)
- Análise de convergência
- Discussão: Observações e interpretações

### 5. Conclusão
- Síntese dos resultados
- Diferenças práticas observadas Q-Learning vs SARSA
- Limitações e trabalhos futuros
- Lições aprendidas

### 6. Referências
- Sutton & Barto (2018)
- Documentação Gymnasium
- Material do curso

---

## 🐛 Troubleshooting

### Problema: "Import matplotlib could not be resolved"
**Solução**: Instale as dependências:
```bash
pip install -r requirements.txt
```

### Problema: Treinamento muito lento
**Possíveis causas:**
- Renderização habilitada (deve ser `render_mode=None` no treino)
- Máquina lenta
- Muitos bins (3600 estados é OK, mas 10k+ pode ser lento)

**Solução**: Reduzir número de episódios ou bins temporariamente para teste.

### Problema: Agentes não convergem
**Possíveis causas:**
- Taxa de aprendizado muito alta/baixa
- Epsilon não está decaindo
- Bugs na lógica de atualização

**Debug**: Imprimir valores Q durante treino, verificar se estão mudando.

---

## 🎓 Pontos-Chave para Apresentação

1. **Mostre o gráfico** (`comparison_result.png`) e explique as curvas
2. **Demonstre diferença no código**: Loop Q-Learning vs SARSA lado a lado
3. **Execute `evaluate.py`** (se criar) para mostrar agente treinado em ação
4. **Explique on-policy vs off-policy** com analogia simples
5. **Conecte com material teórico**: Sutton & Barto, equações de Bellman

---

## ✅ Checklist de Entrega

- [ ] Código completo e funcional
- [ ] `comparison_result.png` gerado
- [ ] README.md atualizado (parte c)
- [ ] Relatório escrito (estrutura acima)
- [ ] Código comentado e limpo
- [ ] Apresentação preparada (slides opcionais)
- [ ] Teste final: `python comparison.py` executa sem erros

---

## 📚 Recursos Adicionais

### Referências Principais
- **Sutton & Barto (2018)**: Capítulos 6 (TD Learning) e 10 (On-policy vs Off-policy)
- **Gymnasium Docs**: https://gymnasium.farama.org/environments/classic_control/cart_pole/

### Conceitos-Chave para Revisar
- Processo de Decisão de Markov (MDP)
- Equação de Bellman
- Diferença Temporal (TD Error)
- Política epsilon-greedy
- Exploração vs Exploração (Exploration-Exploitation Trade-off)

---

## 🚀 Otimizações Implementadas (Fases 1A, 1B, 2)

### Resumo das Melhorias

**Performance Final (12,000 episódios):**
- Q-Learning: 21.87 → **114.89 timesteps** (+425%, 5.25x)
- SARSA: 20.85 → **113.32 timesteps** (+444%, 5.43x)

### Fase 1A: Epsilon Adaptativo
```python
EPSILON_DECAY = 0.9995  # Antes: 0.998
EPSILON_MIN = 0.05      # Antes: 0.01
```
**Impacto**: Exploração prolongada até episódio ~9,200, consolidando política ótima.

### Fase 1B: Reward Shaping
```python
# Adicionado em train.py
stability_bonus = 0.2 * (1.0 - abs(pole_angle) / 0.209)
velocity_penalty = -0.05 * (abs(cart_vel) + abs(pole_vel))
reward = reward + stability_bonus + velocity_penalty
```
**Impacto**: Guia aprendizado para estados estáveis (θ≈0) e movimentos suaves.

### Fase 2: Binning Adaptativo Não-Uniforme
```python
# Adicionado em environment.py
def adaptive_linspace(start, end, num, concentration=2.0):
    uniform = np.linspace(-1, 1, num)
    concentrated = np.sign(uniform) * np.abs(uniform) ** concentration
    return critical_center + concentrated * half_range

# 8×8×12×12 = 9,216 estados (antes: 3,600)
cart_pos: 8 bins (concentration=1.5)
cart_vel: 8 bins (concentration=2.0)
pole_angle: 12 bins (concentration=2.5)  ← Mais crítico
pole_vel: 12 bins (concentration=2.0)
```
**Impacto**: Maior resolução em regiões críticas (θ≈0), melhor controle fino.

### Resultados Comparativos

| Fase | Q-Learning | SARSA | Exploração Q | Exploração S |
|------|-----------|-------|--------------|--------------|
| Baseline | 21.87 ts | 20.85 ts | 6.92% | 8.81% |
| Fase 0 | 49.51 ts | 39.77 ts | 13.25% | 10.22% |
| **Final** | **114.89 ts** | **113.32 ts** | **14.59%** | **14.89%** |

**Máximos Atingidos:**
- Q-Learning: 331 timesteps
- SARSA: 436 timesteps

---

**Boa sorte com o relatório! 🚀**

*Este documento foi criado para facilitar a continuidade do trabalho e escrita do relatório final. Qualquer dúvida, consulte o código-fonte com comentários detalhados.*
