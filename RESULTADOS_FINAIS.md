# Resultados Finais - CartPole RL

## Performance Obtida

| Algoritmo | Baseline | Final | Ganho |
|-----------|----------|-------|-------|
| **Q-Learning** | 21.87 ts | 114.89 ts | **+425%** (5.25x) |
| **SARSA** | 20.85 ts | 113.32 ts | **+444%** (5.43x) |

**Máximos atingidos:**
- Q-Learning: 331 timesteps
- SARSA: 436 timesteps

---

## Otimizações Implementadas

### Fase 1A: Epsilon Adaptativo
- `EPSILON_DECAY`: 0.995 → 0.9995
- `EPSILON_MIN`: 0.01 → 0.05
- Exploração prolongada até episódio ~9,200

### Fase 1B: Reward Shaping
- Bônus de estabilidade: `0.2 * (1.0 - |θ|/0.209)`
- Penalidade de velocidade: `-0.05 * (|v_cart| + |v_pole|)`

### Fase 2: Binning Adaptativo
- Estados: 3,600 → 9,216 (8×8×12×12)
- Concentração não-uniforme em regiões críticas (θ≈0)
- Maior resolução para controle fino

---

## Configuração Final

```python
ALPHA = 0.2
GAMMA = 0.99
EPSILON_DECAY = 0.9995
EPSILON_MIN = 0.05
EPISODES = 12000
STATE_SHAPE = (8, 8, 12, 12)
```

---

## Arquivos Principais

**Código:**
- `manual_control.py` - Controle humano (partes a/b)
- `environment.py` - Discretização adaptativa
- `agents.py` - Q-Learning e SARSA
- `train.py` - Loop de treinamento
- `comparison.py` - Script principal
- `evaluate.py` - Teste visual

**Documentação:**
- `README.md` - Documentação acadêmica
- `IMPLEMENTATION_NOTES.md` - Guia técnico

**Resultados:**
- `qlearning_qtable.pkl` - Q-table treinada
- `sarsa_qtable.pkl` - Q-table treinada  
- `comparison_result.png` - Curvas de aprendizado

---

## Como Executar

```bash
# Treinar agentes (12k episódios, ~25 min)
python comparison.py

# Testar visualmente
python evaluate.py --agent qlearning --episodes 10
python evaluate.py --agent sarsa --episodes 10
```

---

## Conclusão

✅ Ambos algoritmos alcançaram performance similar (~115 timesteps)  
✅ Q-Learning mais estável (σ=28.67)  
✅ SARSA maior variância (σ=55.04), picos superiores  
✅ Implementação completa e otimizada com sucesso
