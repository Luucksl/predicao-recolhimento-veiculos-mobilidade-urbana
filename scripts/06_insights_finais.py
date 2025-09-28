#!/usr/bin/env python3
"""
Insights Finais - TCC Predição de Recolhimento
==============================================

Script para gerar análises adicionais e insights finais:
- Análise de complexidade vs performance
- Correlação entre modelos
- Análise de overfitting
- Métricas consolidadas detalhadas
- Visualizações complementares
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Usar backend sem interface gráfica
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from datetime import datetime
import joblib

# Configuração
warnings.filterwarnings('ignore')
plt.ioff()  # Desabilitar modo interativo
plt.style.use('default')
sns.set_palette("husl")

# Paths
OUTPUT_DIR = Path("outputs")
INSIGHTS_DIR = OUTPUT_DIR / "insights_finais"
INSIGHTS_DIR.mkdir(exist_ok=True)

def load_all_metrics():
    """Carrega todas as métricas dos modelos"""
    
    print("Carregando métricas de todos os modelos...")
    
    # Carregar métricas principais
    try:
        df_metricas = pd.read_csv("comparacao/metricas_todos_modelos.csv")
        print(f"[OK] Métricas carregadas: {len(df_metricas)} modelos")
    except FileNotFoundError:
        print("[ERRO] Arquivo de métricas não encontrado")
        return None
    
    # Carregar métricas por categoria
    categorias = {
        'arvore': 'arvore/comparacao_arvores.csv',
        'xgboost': 'xgboost/comparacao_xgboost.csv', 
        'ensemble': 'ensemble/comparacao_ensemble.csv',
        'redes_neurais': 'redes_neurais/comparacao_redes_neurais.csv'
    }
    
    dfs_categoria = {}
    for cat, arquivo in categorias.items():
        try:
            df = pd.read_csv(arquivo)
            dfs_categoria[cat] = df
            print(f"[OK] {cat}: {len(df)} modelos")
        except FileNotFoundError:
            print(f"[AVISO] {cat}: arquivo não encontrado")
    
    return df_metricas, dfs_categoria

def analyze_complexity_performance(df):
    """Análise de complexidade vs performance"""
    
    print("\nAnalisando complexidade vs performance...")
    
    # Calcular correlação entre complexidade e performance
    corr_complexity_auc = df['Complexidade'].corr(df['AUC'])
    corr_complexity_gini = df['Complexidade'].corr(df['Gini'])
    
    # Análise por faixas de complexidade
    df['Complexidade_Group'] = pd.cut(df['Complexidade'], 
                                     bins=[0, 5, 10, 15, 20], 
                                     labels=['Baixa', 'Média', 'Alta', 'Muito Alta'])
    
    complexity_stats = df.groupby('Complexidade_Group').agg({
        'AUC': ['mean', 'std', 'count'],
        'Gini': ['mean', 'std'],
        'Precisão': 'mean',
        'Recall': 'mean'
    }).round(4)
    
    # Gráfico de complexidade vs performance
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # AUC vs Complexidade
    axes[0,0].scatter(df['Complexidade'], df['AUC'], alpha=0.7, s=100)
    axes[0,0].set_xlabel('Complexidade')
    axes[0,0].set_ylabel('AUC')
    axes[0,0].set_title(f'AUC vs Complexidade (r={corr_complexity_auc:.3f})')
    axes[0,0].grid(True, alpha=0.3)
    
    # Adicionar linha de tendência
    z = np.polyfit(df['Complexidade'], df['AUC'], 1)
    p = np.poly1d(z)
    axes[0,0].plot(df['Complexidade'], p(df['Complexidade']), "r--", alpha=0.8)
    
    # Gini vs Complexidade
    axes[0,1].scatter(df['Complexidade'], df['Gini'], alpha=0.7, s=100, color='orange')
    axes[0,1].set_xlabel('Complexidade')
    axes[0,1].set_ylabel('Gini')
    axes[0,1].set_title(f'Gini vs Complexidade (r={corr_complexity_gini:.3f})')
    axes[0,1].grid(True, alpha=0.3)
    
    # Adicionar linha de tendência
    z = np.polyfit(df['Complexidade'], df['Gini'], 1)
    p = np.poly1d(z)
    axes[0,1].plot(df['Complexidade'], p(df['Complexidade']), "r--", alpha=0.8)
    
    # Performance por categoria de complexidade
    complexity_auc = df.groupby('Complexidade_Group')['AUC'].mean()
    axes[1,0].bar(complexity_auc.index, complexity_auc.values, alpha=0.7)
    axes[1,0].set_xlabel('Categoria de Complexidade')
    axes[1,0].set_ylabel('AUC Médio')
    axes[1,0].set_title('AUC Médio por Categoria de Complexidade')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Trade-off Precisão vs Recall
    axes[1,1].scatter(df['Recall'], df['Precisão'], alpha=0.7, s=100, color='green')
    axes[1,1].set_xlabel('Recall')
    axes[1,1].set_ylabel('Precisão')
    axes[1,1].set_title('Trade-off Precisão vs Recall')
    axes[1,1].grid(True, alpha=0.3)
    
    # Adicionar nomes dos modelos
    for i, row in df.iterrows():
        if row['AUC'] > 0.9:  # Apenas modelos com alta performance
            axes[1,1].annotate(row['Modelo'], 
                             (row['Recall'], row['Precisão']),
                             xytext=(5, 5), textcoords='offset points',
                             fontsize=8, alpha=0.8)
    
    plt.tight_layout()
    plt.savefig(INSIGHTS_DIR / 'analise_complexidade_performance.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'corr_complexity_auc': corr_complexity_auc,
        'corr_complexity_gini': corr_complexity_gini,
        'complexity_stats': complexity_stats
    }

def analyze_model_correlations(df):
    """Análise de correlação entre modelos"""
    
    print("\nAnalisando correlações entre modelos...")
    
    # Carregar predições se disponíveis
    predicoes_files = [
        'logit/predicoes_logit.csv',
        'arvore/predicoes_arvores.csv',
        'xgboost/predicoes_xgboost.csv',
        'ensemble/predicoes_ensemble.csv',
        'redes_neurais/predicoes_redes_neurais.csv'
    ]
    
    predicoes_data = {}
    for arquivo in predicoes_files:
        try:
            df_pred = pd.read_csv(arquivo)
            if 'pred_proba' in df_pred.columns:
                modelo_nome = arquivo.split('/')[0].replace('_', ' ').title()
                predicoes_data[modelo_nome] = df_pred['pred_proba']
                print(f"[OK] Predições carregadas: {modelo_nome}")
        except FileNotFoundError:
            print(f"[AVISO] Predições não encontradas: {arquivo}")
    
    if len(predicoes_data) < 2:
        print("[AVISO] Dados insuficientes para análise de correlação")
        return None
    
    # Criar DataFrame de correlações
    df_corr = pd.DataFrame(predicoes_data)
    correlation_matrix = df_corr.corr()
    
    # Gráfico de correlação
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, 
                mask=mask,
                annot=True, 
                cmap='coolwarm', 
                center=0,
                square=True,
                fmt='.3f',
                cbar_kws={"shrink": .8})
    plt.title('Matriz de Correlação entre Predições dos Modelos')
    plt.tight_layout()
    plt.savefig(INSIGHTS_DIR / 'correlacao_modelos_detalhada.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Identificar modelos mais similares e diferentes
    corr_values = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_values.append({
                'Modelo1': correlation_matrix.columns[i],
                'Modelo2': correlation_matrix.columns[j],
                'Correlacao': correlation_matrix.iloc[i, j]
            })
    
    df_corr_pairs = pd.DataFrame(corr_values)
    df_corr_pairs = df_corr_pairs.sort_values('Correlacao', ascending=False)
    
    return {
        'correlation_matrix': correlation_matrix,
        'correlation_pairs': df_corr_pairs
    }

def analyze_overfitting_patterns(df):
    """Análise de padrões de overfitting"""
    
    print("\nAnalisando padrões de overfitting...")
    
    # Identificar possíveis overfitting baseado em performance muito alta
    # e diferenças entre categorias
    df['Possivel_Overfitting'] = (
        (df['AUC'] > 0.95) |  # AUC muito alta
        (df['Acurácia'] > 0.99) |  # Acurácia muito alta
        (df['Precisão'] > 0.95) |  # Precisão muito alta
        (df['Recall'] > 0.95)  # Recall muito alto
    )
    
    # Análise por categoria de modelo
    df['Categoria'] = df['Modelo'].apply(categorize_model)
    
    overfitting_analysis = df.groupby('Categoria').agg({
        'AUC': ['mean', 'max', 'std'],
        'Possivel_Overfitting': 'sum',
        'Complexidade': 'mean'
    }).round(4)
    
    # Gráfico de análise de overfitting
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # AUC por categoria
    categoria_auc = df.groupby('Categoria')['AUC'].agg(['mean', 'std']).reset_index()
    axes[0,0].bar(categoria_auc['Categoria'], categoria_auc['mean'], 
                  yerr=categoria_auc['std'], alpha=0.7, capsize=5)
    axes[0,0].set_xlabel('Categoria de Modelo')
    axes[0,0].set_ylabel('AUC Médio')
    axes[0,0].set_title('AUC por Categoria de Modelo')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Complexidade vs AUC (identificar overfitting) - VERSÃO CORRIGIDA
    colors = ['red' if x else 'blue' for x in df['Possivel_Overfitting']]
    scatter = axes[0,1].scatter(df['Complexidade'], df['AUC'], c=colors, alpha=0.7, s=100, edgecolors='black', linewidth=0.5)
    
    # Adicionar rótulos dos modelos nos pontos principais
    for i, modelo in enumerate(df['Modelo']):
        if df.iloc[i]['AUC'] > 0.85 or df.iloc[i]['Possivel_Overfitting']:
            axes[0,1].annotate(modelo, 
                              (df.iloc[i]['Complexidade'], df.iloc[i]['AUC']),
                              xytext=(3, 3), textcoords='offset points',
                              fontsize=7, alpha=0.9)
    
    axes[0,1].set_xlabel('Complexidade')
    axes[0,1].set_ylabel('AUC')
    axes[0,1].set_title('Análise de Overfitting: AUC vs Complexidade')
    
    # Adicionar legenda
    axes[0,1].scatter([], [], c='red', alpha=0.7, s=100, label='Possível Overfitting')
    axes[0,1].scatter([], [], c='blue', alpha=0.7, s=100, label='Performance Normal')
    axes[0,1].legend(fontsize=8)
    axes[0,1].grid(True, alpha=0.3)
    
    # Distribuição de AUC
    axes[1,0].hist(df['AUC'], bins=15, alpha=0.7, edgecolor='black')
    axes[1,0].axvline(df['AUC'].mean(), color='red', linestyle='--', 
                     label=f'Média: {df["AUC"].mean():.3f}')
    axes[1,0].set_xlabel('AUC')
    axes[1,0].set_ylabel('Frequência')
    axes[1,0].set_title('Distribuição de AUC - Todos os Modelos')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Trade-off Performance vs Estabilidade - VERSÃO CORRIGIDA
    df['Estabilidade'] = 1 / (df['AUC'] * df['Complexidade'] + 1)
    scatter2 = axes[1,1].scatter(df['AUC'], df['Estabilidade'], alpha=0.7, s=100, color='purple', edgecolors='black', linewidth=0.5)
    
    # Adicionar rótulos dos top 5 modelos
    top_models = df.nlargest(5, 'AUC')
    for idx, row in top_models.iterrows():
        axes[1,1].annotate(row['Modelo'], 
                          (row['AUC'], row['Estabilidade']),
                          xytext=(3, 3), textcoords='offset points',
                          fontsize=7, alpha=0.9, fontweight='bold')
    
    axes[1,1].set_xlabel('AUC')
    axes[1,1].set_ylabel('Índice de Estabilidade')
    axes[1,1].set_title('Trade-off Performance vs Estabilidade')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(INSIGHTS_DIR / 'analise_overfitting.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'overfitting_analysis': overfitting_analysis,
        'overfitting_count': df['Possivel_Overfitting'].sum()
    }

def categorize_model(modelo_nome):
    """Categoriza modelo baseado no nome"""
    modelo_lower = modelo_nome.lower()
    
    if 'logit' in modelo_lower or 'logistic' in modelo_lower:
        return 'Regressão Logística'
    elif 'arvore' in modelo_lower or 'tree' in modelo_lower:
        return 'Árvores de Decisão'
    elif 'xgboost' in modelo_lower:
        return 'XGBoost'
    elif 'random' in modelo_lower or 'forest' in modelo_lower:
        return 'Random Forest'
    elif 'mlp' in modelo_lower or 'lstm' in modelo_lower or 'neural' in modelo_lower:
        return 'Redes Neurais'
    elif 'voting' in modelo_lower or 'stacking' in modelo_lower or 'bagging' in modelo_lower or 'boosting' in modelo_lower:
        return 'Ensemble'
    else:
        return 'Outros'

def generate_consolidated_metrics(df):
    """Gera métricas consolidadas detalhadas"""
    
    print("\nGerando métricas consolidadas...")
    
    # Estatísticas gerais
    stats_gerais = {
        'Total_Modelos': len(df),
        'AUC_Medio': df['AUC'].mean(),
        'AUC_Desvio': df['AUC'].std(),
        'AUC_Maximo': df['AUC'].max(),
        'AUC_Minimo': df['AUC'].min(),
        'AUC_Amplitude': df['AUC'].max() - df['AUC'].min(),
        'Gini_Medio': df['Gini'].mean(),
        'Precisao_Media': df['Precisão'].mean(),
        'Recall_Medio': df['Recall'].mean(),
        'F1_Medio': df['F1-Score'].mean()
    }
    
    # Análise por categoria
    df['Categoria'] = df['Modelo'].apply(categorize_model)
    stats_categoria = df.groupby('Categoria').agg({
        'AUC': ['count', 'mean', 'std', 'max', 'min'],
        'Gini': ['mean', 'std'],
        'Precisão': 'mean',
        'Recall': 'mean',
        'F1-Score': 'mean',
        'Complexidade': 'mean'
    }).round(4)
    
    # Top 5 modelos por métrica
    top_auc = df.nlargest(5, 'AUC')[['Modelo', 'AUC', 'Gini', 'Categoria']]
    top_gini = df.nlargest(5, 'Gini')[['Modelo', 'AUC', 'Gini', 'Categoria']]
    top_precision = df.nlargest(5, 'Precisão')[['Modelo', 'Precisão', 'Recall', 'Categoria']]
    top_recall = df.nlargest(5, 'Recall')[['Modelo', 'Precisão', 'Recall', 'Categoria']]
    
    # Salvar métricas consolidadas
    with open(INSIGHTS_DIR / 'metricas_consolidadas.txt', 'w', encoding='utf-8') as f:
        f.write("MÉTRICAS CONSOLIDADAS - TCC PREDIÇÃO DE RECOLHIMENTO\n")
        f.write("="*60 + "\n\n")
        
        f.write("ESTATÍSTICAS GERAIS:\n")
        f.write("-" * 30 + "\n")
        for key, value in stats_gerais.items():
            f.write(f"{key}: {value:.4f}\n")
        
        f.write(f"\nANÁLISE POR CATEGORIA:\n")
        f.write("-" * 30 + "\n")
        f.write(stats_categoria.to_string())
        
        f.write(f"\n\nTOP 5 MODELOS POR AUC:\n")
        f.write("-" * 30 + "\n")
        f.write(top_auc.to_string(index=False))
        
        f.write(f"\n\nTOP 5 MODELOS POR GINI:\n")
        f.write("-" * 30 + "\n")
        f.write(top_gini.to_string(index=False))
        
        f.write(f"\n\nTOP 5 MODELOS POR PRECISÃO:\n")
        f.write("-" * 30 + "\n")
        f.write(top_precision.to_string(index=False))
        
        f.write(f"\n\nTOP 5 MODELOS POR RECALL:\n")
        f.write("-" * 30 + "\n")
        f.write(top_recall.to_string(index=False))
    
    return {
        'stats_gerais': stats_gerais,
        'stats_categoria': stats_categoria,
        'top_auc': top_auc,
        'top_gini': top_gini,
        'top_precision': top_precision,
        'top_recall': top_recall
    }

def generate_final_report(df, complexity_analysis, correlation_analysis, overfitting_analysis, consolidated_metrics):
    """Gera relatório final consolidado"""
    
    print("\nGerando relatório final...")
    
    report_path = INSIGHTS_DIR / 'relatorio_insights_finais.md'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Relatório de Insights Finais - TCC Predição de Recolhimento\n\n")
        f.write(f"**Data de Geração:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n")
        
        f.write("## Resumo Executivo\n\n")
        f.write(f"Análise completa de {len(df)} modelos de Machine Learning para predição de recolhimento de veículos.\n\n")
        
        f.write("### Principais Descobertas:\n")
        f.write(f"- **Melhor Modelo:** {df.loc[df['AUC'].idxmax(), 'Modelo']} (AUC: {df['AUC'].max():.4f})\n")
        f.write(f"- **AUC Médio:** {df['AUC'].mean():.4f} ± {df['AUC'].std():.4f}\n")
        f.write(f"- **Amplitude de Performance:** {df['AUC'].max() - df['AUC'].min():.4f}\n")
        f.write(f"- **Correlação Complexidade-AUC:** {complexity_analysis['corr_complexity_auc']:.3f}\n")
        
        if overfitting_analysis:
            f.write(f"- **Possíveis Overfitting:** {overfitting_analysis['overfitting_count']} modelos\n")
        
        f.write("\n## Análise de Complexidade vs Performance\n\n")
        f.write(f"A correlação entre complexidade e performance é {complexity_analysis['corr_complexity_auc']:.3f}, ")
        if abs(complexity_analysis['corr_complexity_auc']) < 0.3:
            f.write("indicando que complexidade não é o principal fator de performance.\n\n")
        else:
            f.write("indicando relação moderada entre complexidade e performance.\n\n")
        
        f.write("### Estatísticas por Categoria de Complexidade:\n")
        f.write(complexity_analysis['complexity_stats'].to_string())
        f.write("\n\n")
        
        if correlation_analysis:
            f.write("## Análise de Correlação entre Modelos\n\n")
            f.write("### Modelos Mais Similares:\n")
            top_similar = correlation_analysis['correlation_pairs'].head(3)
            for _, row in top_similar.iterrows():
                f.write(f"- {row['Modelo1']} ↔ {row['Modelo2']}: {row['Correlacao']:.3f}\n")
            
            f.write("\n### Modelos Mais Diferentes:\n")
            bottom_different = correlation_analysis['correlation_pairs'].tail(3)
            for _, row in bottom_different.iterrows():
                f.write(f"- {row['Modelo1']} ↔ {row['Modelo2']}: {row['Correlacao']:.3f}\n")
            f.write("\n")
        
        f.write("## Recomendações Operacionais\n\n")
        f.write("### Para Produção:\n")
        f.write(f"1. **Modelo Principal:** {df.loc[df['AUC'].idxmax(), 'Modelo']} - melhor performance geral\n")
        
        # Encontrar melhor trade-off precisão-recall
        df['Precision_Recall_Harmonic'] = 2 * (df['Precisão'] * df['Recall']) / (df['Precisão'] + df['Recall'])
        best_balance = df.loc[df['Precision_Recall_Harmonic'].idxmax()]
        f.write(f"2. **Melhor Balanceado:** {best_balance['Modelo']} - melhor equilíbrio precisão-recall\n")
        
        # Modelo mais simples com boa performance
        df['Performance_Simplicity'] = df['AUC'] / df['Complexidade']
        best_simple = df.loc[df['Performance_Simplicity'].idxmax()]
        f.write(f"3. **Mais Eficiente:** {best_simple['Modelo']} - melhor custo-benefício\n")
        
        f.write("\n### Para Interpretabilidade:\n")
        f.write("1. **Árvores de Decisão** oferecem melhor interpretabilidade\n")
        f.write("2. **Regressão Logística** para análise de coeficientes\n")
        f.write("3. **XGBoost** para feature importance detalhada\n")
        
        f.write("\n## Visualizações Geradas\n\n")
        f.write("- `analise_complexidade_performance.png` - Análise de complexidade vs performance\n")
        f.write("- `correlacao_modelos_detalhada.png` - Matriz de correlação entre modelos\n")
        f.write("- `analise_overfitting.png` - Análise de padrões de overfitting\n")
        f.write("- `metricas_consolidadas.txt` - Métricas detalhadas em texto\n")
        
        f.write("\n## Próximos Passos\n\n")
        f.write("1. Implementar modelo principal em produção\n")
        f.write("2. Configurar monitoramento de performance\n")
        f.write("3. Estabelecer pipeline de retreinamento\n")
        f.write("4. Validar com dados de holdout\n")
    
    print(f"[OK] Relatório final salvo em: {report_path}")

def main():
    """Função principal"""
    
    print("="*80)
    print("INSIGHTS FINAIS - TCC PREDIÇÃO DE RECOLHIMENTO")
    print("="*80)
    print(f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    
    # Carregar dados
    df_metricas, dfs_categoria = load_all_metrics()
    if df_metricas is None:
        print("❌ Não foi possível carregar as métricas")
        return
    
    # Análises
    print("\n" + "="*60)
    print("EXECUTANDO ANÁLISES")
    print("="*60)
    
    # 1. Análise de complexidade vs performance
    complexity_analysis = analyze_complexity_performance(df_metricas)
    
    # 2. Análise de correlação entre modelos
    correlation_analysis = analyze_model_correlations(df_metricas)
    
    # 3. Análise de overfitting
    overfitting_analysis = analyze_overfitting_patterns(df_metricas)
    
    # 4. Métricas consolidadas
    consolidated_metrics = generate_consolidated_metrics(df_metricas)
    
    # 5. Relatório final
    generate_final_report(df_metricas, complexity_analysis, correlation_analysis, 
                         overfitting_analysis, consolidated_metrics)
    
    print("\n" + "="*60)
    print("ANÁLISE CONCLUÍDA")
    print("="*60)
    print(f"[OK] Insights salvos em: {INSIGHTS_DIR}")
    print("[OK] Relatório final gerado")
    print("[OK] Visualizações criadas")
    
    print(f"\nArquivos gerados:")
    for arquivo in INSIGHTS_DIR.glob("*"):
        print(f"  - {arquivo.name}")

if __name__ == "__main__":
    main()
