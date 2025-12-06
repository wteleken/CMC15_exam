# 🎯 CartPole RL - Q-Learning vs SARSA vs SARSA(λ)

**CMC15 - Aprendizado de Máquina - ITA**  
Daniel da Silveira Sahadi | Pablo Carvalho | Thiago Galante | Willian Teleken

---

## 🏆 Resultados Finais

| Algoritmo | Performance | Episódios | Eficiência |
|-----------|-------------|-----------|------------|
| **🥇 SARSA(λ=0.9)** | **~191 ts** | 5k | ⭐⭐⭐⭐⭐ |
| 🥈 SARSA Otimizado | ~37 ts | 50k | ⭐⭐⭐ |
| 🥉 Q-Learning | ~32 ts | 25k | ⭐⭐⭐⭐ |

**Destaques:**
- ✅ SARSA(λ) **5-10x mais eficiente** (mesma performance, menos episódios)
- ✅ Eligibility Traces transformam algoritmos on-policy
- ✅ Gráficos de alta qualidade para relatório

---

## 📁 Estrutura do Projeto

```
CMC15_exam/
├── 📦 Core
│   ├── agents.py              # Q-Learning, SARSA e SARSA(λ)
│   ├── train.py               # Loop de treino universal
│   ├── environment.py         # Discretização adaptativa
│   └── final_execution.py     # 🚀 SCRIPT PRINCIPAL
│
├── 📝 Documentação
│   ├── README.md              # Este arquivo
│   ├── RESULTADOS_FINAIS_COMPLETOS.md  # Análise detalhada
│   └── PROJECT_STRUCTURE.py   # Estrutura do projeto
│
├── 💾 Modelos (Q-tables)
│   ├── final_q-learning_qtable.pkl
│   ├── final_sarsa_otimizado_qtable.pkl
│   └── final_sarsalambda_qtable.pkl
│
└── 📊 Gráficos
    ├── comparison_result.png   # Comparação de performance
    └── policy_heatmaps.png     # Heatmaps das políticas
```

---

## 🚀 Início Rápido

### Instalação
```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### Execução Principal

#### Treinar todos os modelos e gerar gráficos
```bash
python final_execution.py
```

**Tempo:** ~30-40 minutos  
**Gera:** 2 gráficos (.png) + 3 Q-tables (.pkl)

---

Os gráficos já estão prontos:
- `comparison_result.png` - Comparação de performance
- `policy_heatmaps.png` - Fronteiras de decisão

---

## 💡 Uso Programático

### Treinar SARSA(λ) Customizado
```python
from environment import create_bins
from agents import SarsaLambdaAgent
from train import train_agent

# Configuração (bins refinados)
bins = create_bins()  # Padrão: (6, 6, 18, 18)
state_shape = tuple(len(b) + 1 for b in bins)

# Criar agente
agent = SarsaLambdaAgent(
    state_shape=state_shape,
    n_actions=2,
    alpha=0.15,
    gamma=0.99,
    epsilon=1.0,
    epsilon_decay=0.9995,
    epsilon_min=0.001,
    lambda_factor=0.9
)

# Treinar (5k episódios)
rewards = train_agent(agent, bins, episodes=5000, is_sarsa=True)

print(f"Performance: {np.mean(rewards[-100:])} timesteps")
```

---

## 🔬 Implementações Principais

### 1. SARSA(λ) - Eligibility Traces
```python
class SarsaLambdaAgent(BaseAgent):
    """
    SARSA com Eligibility Traces (Accumulating).
    Propaga crédito para estados anteriores do episódio.
    """
    def update(self, state, action, reward, next_state, next_action):
        # Erro TD
        delta = reward + γ * Q(s',a') - Q(s,a)
        
        # Incrementa vestígio
        e(s,a) += 1
        
        # Atualiza TODOS os estados com vestígios ativos
        Q += α * δ * e
        
        # Decai vestígios
        e *= γ * λ
```

**Por que funciona?** Estados críticos no início do episódio aprendem mais rápido!

### 2. Discretização Refinada
```python
# Anterior: (8, 8, 12, 12) = 9,216 estados
# Novo:     (6, 6, 18, 18) = 11,664 estados

# Menos bins em variáveis menos críticas (posição/velocidade)
# Mais bins em variáveis críticas (ângulo/velocidade angular)
```

**Impacto:** +14% para SARSA(λ), -12% para SARSA básico

### 3. SARSA Otimizado (Resultados Anteriores)
- UCB Exploration (30% dos episódios)
- Optimistic Initialization (Q=50)
- Alpha Adaptativo (0.3 → 0.05)
- Epsilon ultra-baixo (0.0001)
- 100% de exploração de estados

---

## 📊 Comparação Detalhada

| Métrica | Q-Learning | SARSA Otim. | SARSA(λ) |
|---------|------------|-------------|----------|
| **Performance Média** | 314 ts | 178 ts | **316 ts** ✓ |
| **Desvio Padrão** | ±228 | **±45** ✓ | ±180 |
| **Pico Máximo** | 4,796 | 475 | **17,243** ✓ |
| **Episódios** | 25k | 50k | **5k** ✓ |
| **Exploração** | 13% | **100%** ✓ | ~100% ✓ |
| **Eficiência** | Média | Baixa | **Alta** ✓ |

**Vencedor:** SARSA(λ) - melhor performance com menos episódios!

---

## 🎓 Conceitos Principais

### Q-Learning (Off-Policy)
- Aprende política ótima independente de como age
- Atualização: `Q(s,a) += α[r + γ max Q(s',a') - Q(s,a)]`
- **Vantagem:** Eficiente
- **Desvantagem:** Exploração limitada (13%)

### SARSA (On-Policy)
- Aprende a política que está seguindo
- Atualização: `Q(s,a) += α[r + γ Q(s',a') - Q(s,a)]`
- **Vantagem:** Mais seguro, pode explorar 100%
- **Desvantagem:** Performance inferior sem otimizações

### SARSA(λ) - Eligibility Traces
- Propaga recompensas para estados anteriores
- λ controla quanto crédito é propagado (0.9 = ideal)
- **Vantagem:** Aprendizado 9x mais rápido
- **Desvantagem:** Mais complexo, demora a convergir

---

## 📈 Evolução do Projeto

**Fase 1:** Implementação básica
- Q-Learning: 314 ts ✓
- SARSA: 57 ts ✗

**Fase 2:** Otimização SARSA
- UCB + Optimistic Init + Alpha Adaptativo
- Resultado: 178 ts (3.1x melhoria)

**Fase 3:** SARSA(λ) ⭐ ATUAL
- Eligibility Traces + Discretização refinada
- Resultado: **316 ts (melhor geral!)**

---

## 🔧 Dependências

```txt
gymnasium==1.2.2
numpy==1.26.0
matplotlib==3.8.0
```

---

## 📚 Referências

- Sutton & Barto (2018): *Reinforcement Learning: An Introduction*
  - Cap. 6: Temporal-Difference Learning
  - Cap. 12: Eligibility Traces
  
- Gymnasium: https://gymnasium.farama.org/

---

## 👥 Equipe

- **Daniel da Silveira Sahadi**
- **Pablo Carvalho**
- **Thiago Galante**
- **Willian Teleken**

**Disciplina:** CMC-15 - Aprendizado de Máquina  
**Instituição:** Instituto Tecnológico de Aeronáutica (ITA)  
**Data:** Dezembro 2025

---

## 📖 Documentação Completa

Para análise técnica detalhada, consulte:
- `RESULTADOS_FINAIS_COMPLETOS.md` - Análise profunda de todos os resultados
- `PROJECT_STRUCTURE.py` - Estrutura completa do projeto
