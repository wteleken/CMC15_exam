# 🎯 Resultados Finais - CartPole RL

**CMC15 - 5 de dezembro de 2025**

---

## 📊 Performance Final

| Algoritmo | Média ± DP | Máximo | Exploração |
|-----------|-----------|--------|------------|
| **Q-Learning** | 314 ± 228 ts | 4,796 ts | 13.2% (1,218/9,216) |
| **SARSA Otimizado** | 178 ± 45 ts | 475 ts | **100%** (9,216/9,216) ✓ |
| SARSA Baseline | 57 ± 19 ts | 119 ts | 9.6% (885/9,216) |

**Evolução SARSA**: 57 → 178 ts (**3.1x melhor**) + 100% exploração de estados!

---

## ⚙️ Configurações

### Q-Learning
```python
STATE_SHAPE = (8, 8, 12, 12)  # 9,216 estados
ALPHA = 0.15
EPSILON_DECAY = 0.9995
EPSILON_MIN = 0.001
EPISODES = 25,000
```

### SARSA Otimizado
```python
STATE_SHAPE = (8, 8, 12, 12)
ALPHA = 0.3 → 0.05  # Adaptativo
EPSILON_DECAY = 0.9993
EPSILON_MIN = 0.0001
EPISODES = 50,000
UCB = True  # Primeiros 15k eps
INIT_VALUE = 50.0  # Otimista
```

---

## 🔧 Otimizações Implementadas no SARSA

1. **Inicialização Otimista** (Q=50)
   - Força exploração agressiva no início
   
2. **UCB Exploration** (15k eps)
   - `Q(s,a) + c*sqrt(ln(t)/N(s,a))`
   - Prioriza estados não visitados (bonus infinito)
   
3. **Alpha Adaptativo** (0.3 → 0.05)
   - Aprendizado rápido + refinamento gradual
   
4. **Epsilon Ultra-Baixo** (0.0001)
   - Política 99.99% determinística no final
   
5. **Dobro de Episódios** (50k)
   - Compensa convergência on-policy mais lenta

**Impacto**: 100% exploração + 3x performance!

---

## 📈 Análise

| Métrica | Q-Learning | SARSA | Observação |
|---------|-----------|-------|------------|
| Convergência | ~9k eps | ~15k eps | Q-Learning 1.7x mais rápido |
| Performance | 314 ts | 178 ts | Q-Learning 1.8x superior |
| Estabilidade | ±228 | ±45 | SARSA mais consistente |
| Exploração | 13.2% | 100% | SARSA 7.6x melhor |

### Por que Q-Learning domina?
- **Off-policy**: Aprende política ótima durante exploração
- **Determinístico**: Sem penalidades no CartPole
- **Agressivo**: `max Q(s',a')` converge mais rápido

### Como SARSA melhorou?
- **UCB** >> ε-greedy (exploração inteligente)
- **Inicialização otimista** funciona muito bem
- **Alpha adaptativo** acelera convergência

---

## 📁 Arquivos

```
agents.py              # Q-Learning + SARSA (com UCB)
train.py               # Treino + alpha adaptativo  
environment.py         # Discretização 8×8×12×12
comparison.py          # Script principal
analyze_results.py     # Análise estatística
evaluate.py            # Visualização
qlearning_qtable.pkl   # Melhor Q-Learning (314 ts)
sarsa_qtable.pkl       # Melhor SARSA (178 ts)
```

---

## ✅ Conclusão

**Q-Learning** é ideal para CartPole: 314 ts médios com configuração simples.

**SARSA otimizado** alcançou 178 ts (3x melhor), provando que técnicas avançadas (UCB, inicialização otimista, alpha adaptativo) melhoram significativamente algoritmos on-policy.

**Trade-off**: Q-Learning → performance | SARSA → exploração completa
