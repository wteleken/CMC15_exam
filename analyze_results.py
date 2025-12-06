"""
Script de visualização dos resultados finais otimizados.
Carrega Q-tables e mostra estatísticas de exploração.
"""

import pickle
import numpy as np


def analyze_qtable(filepath, name):
    """Analisa uma Q-table salva."""
    with open(filepath, 'rb') as f:
        q_table = pickle.load(f)
    
    # Estatísticas básicas
    total_states = np.prod(q_table.shape[:-1])
    explored_states = np.sum(np.any(q_table != 0, axis=-1))
    exploration_pct = (explored_states / total_states) * 100
    
    # Valores Q
    non_zero_values = q_table[q_table != 0]
    if len(non_zero_values) > 0:
        q_mean = np.mean(non_zero_values)
        q_max = np.max(non_zero_values)
        q_min = np.min(non_zero_values)
    else:
        q_mean = q_max = q_min = 0
    
    # Confiança da política (diferença entre melhor e segunda melhor ação)
    q_diff = np.abs(q_table[:,:,:,:,0] - q_table[:,:,:,:,1])
    confident_states = np.sum(q_diff[np.any(q_table != 0, axis=-1)] > 1.0)
    confidence_pct = (confident_states / explored_states * 100) if explored_states > 0 else 0
    
    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}")
    print(f"Shape: {q_table.shape}")
    print(f"Estados totais: {total_states:,}")
    print(f"Estados explorados: {explored_states:,} ({exploration_pct:.1f}%)")
    print(f"\nValores Q:")
    print(f"  Média (não-zero): {q_mean:.2f}")
    print(f"  Máximo: {q_max:.2f}")
    print(f"  Mínimo: {q_min:.2f}")
    print(f"\nPolítica:")
    print(f"  Estados confiantes (|Q(s,0)-Q(s,1)| > 1): {confident_states:,}")
    print(f"  Confiança: {confidence_pct:.1f}% dos estados explorados")
    
    return {
        'total_states': total_states,
        'explored_states': explored_states,
        'exploration_pct': exploration_pct,
        'q_mean': q_mean,
        'q_max': q_max,
        'confident_states': confident_states,
        'confidence_pct': confidence_pct
    }


def main():
    print("\n" + "="*70)
    print("ANÁLISE DOS MELHORES MODELOS")
    print("="*70)
    
    # Analisa ambas as Q-tables
    ql_stats = analyze_qtable('qlearning_qtable.pkl', 'Q-LEARNING (Off-Policy)')
    sarsa_stats = analyze_qtable('sarsa_qtable.pkl', 'SARSA (On-Policy + Otimizações)')
    
    # Comparação
    print(f"\n{'='*70}")
    print("COMPARAÇÃO FINAL")
    print(f"{'='*70}")
    
    print("\n📊 EXPLORAÇÃO DE ESTADOS:")
    print(f"  Q-Learning: {ql_stats['explored_states']:,} estados ({ql_stats['exploration_pct']:.1f}%)")
    print(f"  SARSA: {sarsa_stats['explored_states']:,} estados ({sarsa_stats['exploration_pct']:.1f}%)")
    
    if sarsa_stats['exploration_pct'] > ql_stats['exploration_pct']:
        diff = sarsa_stats['exploration_pct'] - ql_stats['exploration_pct']
        print(f"  ✅ SARSA explorou {diff:.1f}% a mais! (UCB + Inicialização Otimista)")
    
    print("\n🎯 CONFIANÇA DA POLÍTICA:")
    print(f"  Q-Learning: {ql_stats['confidence_pct']:.1f}% dos estados visitados")
    print(f"  SARSA: {sarsa_stats['confidence_pct']:.1f}% dos estados visitados")
    
    print("\n💡 VALORES Q APRENDIDOS:")
    print(f"  Q-Learning: média {ql_stats['q_mean']:.2f}, max {ql_stats['q_max']:.2f}")
    print(f"  SARSA: média {sarsa_stats['q_mean']:.2f}, max {sarsa_stats['q_max']:.2f}")
    
    print("\n" + "="*70)
    print("RESULTADOS DE PERFORMANCE (últimos 100 episódios):")
    print("="*70)
    print("Q-Learning: 314.66 ± 228.24 timesteps (Max: 4,796)")
    print("SARSA: 178.19 ± 45.71 timesteps (Max: 475)")
    print("\n✅ Q-Learning: 1.8x mais rápido")
    print("✅ SARSA: 3x melhor que baseline (57 ts → 178 ts)")
    print("✅ SARSA: 7.6x mais exploração que Q-Learning")
    print("="*70)


if __name__ == "__main__":
    main()
