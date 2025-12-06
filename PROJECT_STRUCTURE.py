"""
═══════════════════════════════════════════════════════════════════════
    ESTRUTURA DO PROJETO - CARTPOLE RL
═══════════════════════════════════════════════════════════════════════
"""

ARQUIVOS_CORE = {
    'agents.py': 'Q-Learning e SARSA com UCB exploration',
    'train.py': 'Loop de treino + alpha adaptativo + UCB',
    'environment.py': 'Discretização 8×8×12×12 = 9,216 estados',
    'comparison.py': 'Script principal (treina ambos algoritmos)',
}

ARQUIVOS_ANALISE = {
    'analyze_results.py': 'Análise estatística das Q-tables',
    'evaluate.py': 'Visualização dos agentes treinados',
}

MODELOS_TREINADOS = {
    'qlearning_qtable.pkl': 'Melhor Q-Learning (314 ± 228 ts)',
    'sarsa_qtable.pkl': 'Melhor SARSA (178 ± 45 ts)',
}

DOCUMENTACAO = {
    'README.md': 'Documentação principal + análise teórica',
    'RESULTADO_FINAL_OTIMIZADO.md': 'Resultados detalhados',
    'requirements.txt': 'Dependências Python',
}

VISUALIZACAO = {
    'comparison_result.png': 'Gráfico comparativo final',
}

# Ordem de execução recomendada
ORDEM_EXECUCAO = [
    '1. python comparison.py      # Treinar modelos (~105 min)',
    '2. python analyze_results.py # Analisar estatísticas',
    '3. python evaluate.py         # Visualizar agentes',
]

def print_structure():
    print("\n╔═══════════════════════════════════════════════════════════════════╗")
    print("║              ESTRUTURA DO PROJETO - CARTPOLE RL                   ║")
    print("╚═══════════════════════════════════════════════════════════════════╝\n")
    
    print("📦 CÓDIGO PRINCIPAL")
    for file, desc in ARQUIVOS_CORE.items():
        print(f"  • {file:<25} {desc}")
    
    print("\n📊 ANÁLISE E VISUALIZAÇÃO")
    for file, desc in ARQUIVOS_ANALISE.items():
        print(f"  • {file:<25} {desc}")
    
    print("\n💾 MODELOS TREINADOS")
    for file, desc in MODELOS_TREINADOS.items():
        print(f"  • {file:<25} {desc}")
    
    print("\n📝 DOCUMENTAÇÃO")
    for file, desc in DOCUMENTACAO.items():
        print(f"  • {file:<25} {desc}")
    
    print("\n🎨 VISUALIZAÇÃO")
    for file, desc in VISUALIZACAO.items():
        print(f"  • {file:<25} {desc}")
    
    print("\n🚀 ORDEM DE EXECUÇÃO")
    for step in ORDEM_EXECUCAO:
        print(f"  {step}")
    
    print("\n" + "═"*71)
    print("Total: 10 arquivos principais + modelos treinados")
    print("═"*71 + "\n")

if __name__ == "__main__":
    print_structure()
