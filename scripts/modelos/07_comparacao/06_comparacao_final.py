# -*- coding: utf-8 -*-
"""
UNIVERSIDADE DE SÃO PAULO
MBA DATA SCIENCE & ANALYTICS USP/ESALQ
PREDIÇÃO DE RECOLHIMENTO DE VEÍCULOS - COMPARAÇÃO FINAL
TCC - Pipeline ML Motto

@author: [Seu Nome]
@date: 2025
"""

#%% Importação dos pacotes

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo
import matplotlib.pyplot as plt
from PIL import Image
import io

def salvar_figura_segura(figura, nome_arquivo, dpi=300):
    """Salva figura de forma robusta, evitando erros do PIL"""
    try:
        figura.savefig(nome_arquivo, dpi=dpi, bbox_inches='tight')
        plt.close(figura)
        print(f"Figura salva: {nome_arquivo}")
    except Exception as e:
        print(f"Erro ao salvar {nome_arquivo}: {e}")
        # Tentar salvar em formato diferente
        try:
            figura.savefig(nome_arquivo.replace('.png', '.pdf'), bbox_inches='tight')
            plt.close(figura)
            print(f"Figura salva como PDF: {nome_arquivo.replace('.png', '.pdf')}")
        except Exception as e2:
            print(f"Erro ao salvar como PDF: {e2}")
            plt.close(figura)
from sklearn.metrics import roc_auc_score, roc_curve, auc
import joblib
import warnings
warnings.filterwarnings('ignore')

#%% Configurações de visualização

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

#%% Funções auxiliares

def load_model_results():
    """Carregar resultados de todos os modelos"""
    resultados = {}
    
    # Carregar predições de cada tipo de modelo
    try:
        # Logit
        pred_logit = pd.read_csv('../logit/predicoes_logit.csv')
        resultados['Logit Simples'] = {
            'y_true': pred_logit['recolhimento_evento'],
            'y_pred_proba': pred_logit['phat_simples']
        }
        resultados['Logit Stepwise'] = {
            'y_true': pred_logit['recolhimento_evento'],
            'y_pred_proba': pred_logit['phat_stepwise']
        }
    except:
        print("Aviso: Não foi possível carregar resultados do Logit")
    
    try:
        # Árvores
        pred_arvore = pd.read_csv('../arvore/predicoes_arvores.csv')
        resultados['Árvore Simples'] = {
            'y_true': pred_arvore['y_true'],
            'y_pred_proba': pred_arvore['y_pred_proba_simples']
        }
        resultados['Árvore Podada'] = {
            'y_true': pred_arvore['y_true'],
            'y_pred_proba': pred_arvore['y_pred_proba_podada']
        }
        resultados['Árvore Otimizada'] = {
            'y_true': pred_arvore['y_true'],
            'y_pred_proba': pred_arvore['y_pred_proba_otimizada']
        }
    except:
        print("Aviso: Não foi possível carregar resultados das Árvores")
    
    try:
        # XGBoost
        pred_xgb = pd.read_csv('../xgboost/predicoes_xgboost.csv')
        resultados['XGBoost Simples'] = {
            'y_true': pred_xgb['y_true'],
            'y_pred_proba': pred_xgb['y_pred_proba_simples']
        }
        resultados['XGBoost Early'] = {
            'y_true': pred_xgb['y_true'],
            'y_pred_proba': pred_xgb['y_pred_proba_early']
        }
        resultados['XGBoost Otimizado'] = {
            'y_true': pred_xgb['y_true'],
            'y_pred_proba': pred_xgb['y_pred_proba_otimizado']
        }
        resultados['XGBoost Temporal'] = {
            'y_true': pred_xgb['y_true'],
            'y_pred_proba': pred_xgb['y_pred_proba_temporal']
        }
    except:
        print("Aviso: Não foi possível carregar resultados do XGBoost")
    
    try:
        # Ensemble
        pred_ensemble = pd.read_csv('../ensemble/predicoes_ensemble.csv')
        resultados['Voting Soft'] = {
            'y_true': pred_ensemble['y_true'],
            'y_pred_proba': pred_ensemble['Voting Soft']
        }
        resultados['Stacking'] = {
            'y_true': pred_ensemble['y_true'],
            'y_pred_proba': pred_ensemble['Stacking']
        }
        resultados['Random Forest'] = {
            'y_true': pred_ensemble['y_true'],
            'y_pred_proba': pred_ensemble['Random Forest']
        }
    except:
        print("Aviso: Não foi possível carregar resultados do Ensemble")
    
    try:
        # Redes Neurais
        pred_rn = pd.read_csv('../redes_neurais/predicoes_redes_neurais.csv')
        resultados['MLP Simples'] = {
            'y_true': pred_rn['y_true'],
            'y_pred_proba': pred_rn['y_pred_proba_mlp_simples']
        }
        resultados['MLP TensorFlow'] = {
            'y_true': pred_rn['y_true'],
            'y_pred_proba': pred_rn['y_pred_proba_mlp_tf']
        }
        resultados['LSTM'] = {
            'y_true': pred_rn['y_true'],
            'y_pred_proba': pred_rn['y_pred_proba_lstm']
        }
    except:
        print("Aviso: Não foi possível carregar resultados das Redes Neurais")
    
    return resultados

def calculate_metrics(resultados):
    """Calcular métricas para todos os modelos"""
    metricas = []

    for nome, dados in resultados.items():
        y_true = dados['y_true']
        y_pred_proba = dados['y_pred_proba']

        # Calcular AUC
        auc = roc_auc_score(y_true, y_pred_proba)
        gini = 2 * auc - 1

        # Calcular outras métricas
        y_pred = (y_pred_proba >= 0.5).astype(int)

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        # Calcular complexidade baseada no tipo de modelo
        complexidade = calculate_complexity(nome)

        metricas.append({
            'Modelo': nome,
            'AUC': auc,
            'Gini': gini,
            'Acurácia': accuracy,
            'Precisão': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Complexidade': complexidade
        })

    return pd.DataFrame(metricas)

def calculate_complexity(modelo_nome):
    """Calcula complexidade baseada no tipo de modelo"""
    modelo_lower = modelo_nome.lower()
    
    if 'logit' in modelo_lower or 'logistic' in modelo_lower:
        return 1
    elif 'arvore' in modelo_lower and 'simples' in modelo_lower:
        return 3
    elif 'arvore' in modelo_lower and 'podada' in modelo_lower:
        return 4
    elif 'arvore' in modelo_lower and 'otimizada' in modelo_lower:
        return 5
    elif 'xgboost' in modelo_lower and 'simples' in modelo_lower:
        return 6
    elif 'xgboost' in modelo_lower and 'early' in modelo_lower:
        return 7
    elif 'xgboost' in modelo_lower and 'otimizado' in modelo_lower:
        return 8
    elif 'xgboost' in modelo_lower and 'temporal' in modelo_lower:
        return 9
    elif 'mlp' in modelo_lower and 'simples' in modelo_lower:
        return 10
    elif 'mlp' in modelo_lower and 'tensorflow' in modelo_lower:
        return 11
    elif 'lstm' in modelo_lower:
        return 13
    elif 'voting' in modelo_lower and 'soft' in modelo_lower:
        return 14
    elif 'stacking' in modelo_lower:
        return 15
    elif 'random' in modelo_lower or 'forest' in modelo_lower:
        return 16
    else:
        return 12  # Default para outros modelos

def plot_analise_complexidade(df_metricas):
    """Plotar análise de complexidade vs performance"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # AUC vs Complexidade
    axes[0,0].scatter(df_metricas['Complexidade'], df_metricas['AUC'], alpha=0.7, s=100)
    axes[0,0].set_xlabel('Complexidade')
    axes[0,0].set_ylabel('AUC')
    axes[0,0].set_title('AUC vs Complexidade')
    axes[0,0].grid(True, alpha=0.3)
    
    # Adicionar linha de tendência
    z = np.polyfit(df_metricas['Complexidade'], df_metricas['AUC'], 1)
    p = np.poly1d(z)
    axes[0,0].plot(df_metricas['Complexidade'], p(df_metricas['Complexidade']), "r--", alpha=0.8)
    
    # Gini vs Complexidade
    axes[0,1].scatter(df_metricas['Complexidade'], df_metricas['Gini'], alpha=0.7, s=100, color='orange')
    axes[0,1].set_xlabel('Complexidade')
    axes[0,1].set_ylabel('Gini')
    axes[0,1].set_title('Gini vs Complexidade')
    axes[0,1].grid(True, alpha=0.3)
    
    # Adicionar linha de tendência
    z = np.polyfit(df_metricas['Complexidade'], df_metricas['Gini'], 1)
    p = np.poly1d(z)
    axes[0,1].plot(df_metricas['Complexidade'], p(df_metricas['Complexidade']), "r--", alpha=0.8)
    
    # Trade-off Precisão vs Recall
    axes[1,0].scatter(df_metricas['Recall'], df_metricas['Precisão'], alpha=0.7, s=100, color='green')
    axes[1,0].set_xlabel('Recall')
    axes[1,0].set_ylabel('Precisão')
    axes[1,0].set_title('Trade-off Precisão vs Recall')
    axes[1,0].grid(True, alpha=0.3)
    
    # Adicionar nomes dos modelos
    for i, row in df_metricas.iterrows():
        if row['AUC'] > 0.9:  # Apenas modelos com alta performance
            axes[1,0].annotate(row['Modelo'], 
                             (row['Recall'], row['Precisão']),
                             xytext=(5, 5), textcoords='offset points',
                             fontsize=8, alpha=0.8)
    
    # Performance por categoria de complexidade
    df_metricas['Complexidade_Group'] = pd.cut(df_metricas['Complexidade'], 
                                             bins=[0, 5, 10, 15, 20], 
                                             labels=['Baixa', 'Média', 'Alta', 'Muito Alta'])
    complexity_auc = df_metricas.groupby('Complexidade_Group')['AUC'].mean()
    axes[1,1].bar(complexity_auc.index, complexity_auc.values, alpha=0.7)
    axes[1,1].set_xlabel('Categoria de Complexidade')
    axes[1,1].set_ylabel('AUC Médio')
    axes[1,1].set_title('AUC Médio por Categoria de Complexidade')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    salvar_figura_segura(plt.gcf(), 'evolucao_complexidade.png')
    
    return fig

def plot_comparacao_completa(df_metricas, resultados):
    """Plotar comparação completa de todos os modelos"""
    
    # Ordenar por AUC
    df_metricas = df_metricas.sort_values('AUC', ascending=False)
    
    plt.figure(figsize=(20, 15))
    
    # 1. Comparação de AUC
    plt.subplot(3, 3, 1)
    bars = plt.bar(range(len(df_metricas)), df_metricas['AUC'], alpha=0.8)
    plt.ylabel('AUC', fontsize=12)
    plt.title('Comparação de AUC', fontsize=14)
    plt.xticks(range(len(df_metricas)), df_metricas['Modelo'], rotation=45, ha='right')
    for i, (bar, auc) in enumerate(zip(bars, df_metricas['AUC'])):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{auc:.3f}', ha='center', va='bottom', fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # 2. Comparação de Gini
    plt.subplot(3, 3, 2)
    bars = plt.bar(range(len(df_metricas)), df_metricas['Gini'], alpha=0.8, color='orange')
    plt.ylabel('Gini', fontsize=12)
    plt.title('Comparação de Gini', fontsize=14)
    plt.xticks(range(len(df_metricas)), df_metricas['Modelo'], rotation=45, ha='right')
    for i, (bar, gini) in enumerate(zip(bars, df_metricas['Gini'])):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{gini:.3f}', ha='center', va='bottom', fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # 3. Curvas ROC
    plt.subplot(3, 3, 3)
    for nome, dados in resultados.items():
        fpr, tpr, _ = roc_curve(dados['y_true'], dados['y_pred_proba'])
        auc = roc_auc_score(dados['y_true'], dados['y_pred_proba'])
        plt.plot(fpr, tpr, linewidth=2, label=f'{nome} (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('1 - Especificidade', fontsize=12)
    plt.ylabel('Sensitividade', fontsize=12)
    plt.title('Curvas ROC', fontsize=14)
    plt.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # 4. Métricas múltiplas
    plt.subplot(3, 3, 4)
    x = np.arange(len(df_metricas))
    width = 0.2
    
    plt.bar(x - 2*width, df_metricas['Acurácia'], width, label='Acurácia', alpha=0.8)
    plt.bar(x - width, df_metricas['Precisão'], width, label='Precisão', alpha=0.8)
    plt.bar(x, df_metricas['Recall'], width, label='Recall', alpha=0.8)
    plt.bar(x + width, df_metricas['F1-Score'], width, label='F1-Score', alpha=0.8)
    
    plt.xlabel('Modelos', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Métricas Múltiplas', fontsize=14)
    plt.xticks(x, df_metricas['Modelo'], rotation=45, ha='right')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 5. Heatmap de correlação entre métricas
    plt.subplot(3, 3, 5)
    metricas_corr = df_metricas[['AUC', 'Gini', 'Acurácia', 'Precisão', 'Recall', 'F1-Score']].corr()
    sns.heatmap(metricas_corr, annot=True, cmap='coolwarm', center=0, square=True, fmt='.3f')
    plt.title('Correlação entre Métricas', fontsize=14)
    
    # 6. Ranking por categoria
    plt.subplot(3, 3, 6)
    categorias = {
        'Logit': df_metricas[df_metricas['Modelo'].str.contains('Logit')]['AUC'].max(),
        'Árvore': df_metricas[df_metricas['Modelo'].str.contains('Árvore')]['AUC'].max(),
        'XGBoost': df_metricas[df_metricas['Modelo'].str.contains('XGBoost')]['AUC'].max(),
        'Ensemble': df_metricas[df_metricas['Modelo'].str.contains('Voting|Stacking|Random Forest')]['AUC'].max(),
        'Redes Neurais': df_metricas[df_metricas['Modelo'].str.contains('MLP|LSTM')]['AUC'].max()
    }
    
    categorias = {k: v for k, v in categorias.items() if not pd.isna(v)}
    plt.bar(categorias.keys(), categorias.values(), alpha=0.8)
    plt.ylabel('Melhor AUC', fontsize=12)
    plt.title('Melhor Modelo por Categoria', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 7. Distribuição de AUC
    plt.subplot(3, 3, 7)
    plt.hist(df_metricas['AUC'], bins=10, alpha=0.7, edgecolor='black')
    plt.axvline(df_metricas['AUC'].mean(), color='red', linestyle='--', label=f'Média: {df_metricas["AUC"].mean():.3f}')
    plt.xlabel('AUC', fontsize=12)
    plt.ylabel('Frequência', fontsize=12)
    plt.title('Distribuição de AUC', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 8. Top 10 modelos
    plt.subplot(3, 3, 8)
    top_10 = df_metricas.head(10)
    bars = plt.barh(range(len(top_10)), top_10['AUC'], alpha=0.8)
    plt.yticks(range(len(top_10)), top_10['Modelo'])
    plt.xlabel('AUC', fontsize=12)
    plt.title('Top 10 Modelos', fontsize=14)
    plt.gca().invert_yaxis()
    for i, (bar, auc) in enumerate(zip(bars, top_10['AUC'])):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{auc:.3f}', ha='left', va='center', fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # 9. Resumo estatístico
    plt.subplot(3, 3, 9)
    plt.axis('off')
    
    resumo_texto = f"""
    RESUMO ESTATÍSTICO
    
    Total de Modelos: {len(df_metricas)}
    
    AUC:
    • Média: {df_metricas['AUC'].mean():.4f}
    • Mediana: {df_metricas['AUC'].median():.4f}
    • Desvio: {df_metricas['AUC'].std():.4f}
    • Máximo: {df_metricas['AUC'].max():.4f}
    • Mínimo: {df_metricas['AUC'].min():.4f}
    
    Gini:
    • Média: {df_metricas['Gini'].mean():.4f}
    • Máximo: {df_metricas['Gini'].max():.4f}
    
    Melhor Modelo:
    {df_metricas.iloc[0]['Modelo']}
    AUC: {df_metricas.iloc[0]['AUC']:.4f}
    """
    
    plt.text(0.1, 0.5, resumo_texto, fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    salvar_figura_segura(plt.gcf(), 'comparacao_completa_modelos.png')
    salvar_figura_segura(plt.gcf(), 'comparacao_plot.png')

def plot_evolucao_complexidade(df_metricas):
    """Plotar evolução da performance vs complexidade"""
    
    # Definir ordem de complexidade
    ordem_complexidade = {
        'Logit Simples': 1,
        'Logit Stepwise': 2,
        'Árvore Simples': 3,
        'Árvore Podada': 4,
        'Árvore Otimizada': 5,
        'XGBoost Simples': 6,
        'XGBoost Early': 7,
        'XGBoost Otimizado': 8,
        'XGBoost Temporal': 9,
        'MLP Simples': 10,
        'MLP TensorFlow': 11,
        'MLP Otimizado': 12,
        'LSTM': 13,
        'Voting Soft': 14,
        'Stacking': 15,
        'Random Forest': 16
    }
    
    # Adicionar coluna de complexidade
    df_metricas['Complexidade'] = df_metricas['Modelo'].map(ordem_complexidade)
    df_metricas = df_metricas.dropna().sort_values('Complexidade')
    
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: AUC vs Complexidade
    plt.subplot(2, 2, 1)
    plt.plot(df_metricas['Complexidade'], df_metricas['AUC'], marker='o', linewidth=2, markersize=8)
    plt.xlabel('Complexidade (Ordem)', fontsize=12)
    plt.ylabel('AUC', fontsize=12)
    plt.title('Evolução da Performance vs Complexidade', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Adicionar labels para pontos importantes
    for i, row in df_metricas.iterrows():
        if row['AUC'] > df_metricas['AUC'].quantile(0.9):  # Top 10%
            plt.annotate(row['Modelo'], (row['Complexidade'], row['AUC']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Subplot 2: Gini vs Complexidade
    plt.subplot(2, 2, 2)
    plt.plot(df_metricas['Complexidade'], df_metricas['Gini'], marker='o', linewidth=2, markersize=8, color='orange')
    plt.xlabel('Complexidade (Ordem)', fontsize=12)
    plt.ylabel('Gini', fontsize=12)
    plt.title('Evolução do Gini vs Complexidade', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Scatter plot AUC vs Gini
    plt.subplot(2, 2, 3)
    scatter = plt.scatter(df_metricas['AUC'], df_metricas['Gini'], 
                         c=df_metricas['Complexidade'], cmap='viridis', s=100, alpha=0.7)
    plt.xlabel('AUC', fontsize=12)
    plt.ylabel('Gini', fontsize=12)
    plt.title('AUC vs Gini (colorido por complexidade)', fontsize=14)
    plt.colorbar(scatter, label='Complexidade')
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Box plot por categoria
    plt.subplot(2, 2, 4)
    
    # Agrupar por categoria
    categorias = {
        'Logit': df_metricas[df_metricas['Modelo'].str.contains('Logit')]['AUC'],
        'Árvore': df_metricas[df_metricas['Modelo'].str.contains('Árvore')]['AUC'],
        'XGBoost': df_metricas[df_metricas['Modelo'].str.contains('XGBoost')]['AUC'],
        'Ensemble': df_metricas[df_metricas['Modelo'].str.contains('Voting|Stacking|Random Forest')]['AUC'],
        'Redes Neurais': df_metricas[df_metricas['Modelo'].str.contains('MLP|LSTM')]['AUC']
    }
    
    categorias = {k: v for k, v in categorias.items() if len(v) > 0}
    
    plt.boxplot(categorias.values(), labels=categorias.keys())
    plt.ylabel('AUC', fontsize=12)
    plt.title('Distribuição de AUC por Categoria', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    salvar_figura_segura(plt.gcf(), 'evolucao_complexidade.png')
    salvar_figura_segura(plt.gcf(), 'comparacao_plot.png')

#%% Carregamento dos dados

print("="*80)
print("COMPARAÇÃO FINAL - TODOS OS MODELOS")
print("="*80)

# Carregar resultados de todos os modelos
resultados = load_model_results()

if not resultados:
    print("Erro: Nenhum resultado encontrado. Execute os scripts individuais primeiro.")
    exit()

print(f"Resultados carregados de {len(resultados)} modelos")

#%% Cálculo das métricas

print("\n" + "="*80)
print("CÁLCULO DAS MÉTRICAS")
print("="*80)

df_metricas = calculate_metrics(resultados)

print("Métricas calculadas para todos os modelos:")
print(df_metricas.head(10))

#%% Análise estatística

print("\n" + "="*80)
print("ANÁLISE ESTATÍSTICA")
print("="*80)

print("Estatísticas descritivas das métricas:")
print(df_metricas[['AUC', 'Gini', 'Acurácia', 'Precisão', 'Recall', 'F1-Score']].describe())

print(f"\nCorrelação entre AUC e Gini: {df_metricas['AUC'].corr(df_metricas['Gini']):.4f}")

#%% Visualizações

print("\n" + "="*80)
print("GERANDO VISUALIZAÇÕES")
print("="*80)

# Comparação completa
plot_comparacao_completa(df_metricas, resultados)

# Evolução da complexidade
plot_evolucao_complexidade(df_metricas)

# Análise de complexidade vs performance
plot_analise_complexidade(df_metricas)

#%% Análise por categoria

print("\n" + "="*80)
print("ANÁLISE POR CATEGORIA")
print("="*80)

categorias = {
    'Logit': df_metricas[df_metricas['Modelo'].str.contains('Logit')],
    'Árvore': df_metricas[df_metricas['Modelo'].str.contains('Árvore')],
    'XGBoost': df_metricas[df_metricas['Modelo'].str.contains('XGBoost')],
    'Ensemble': df_metricas[df_metricas['Modelo'].str.contains('Voting|Stacking|Random Forest')],
    'Redes Neurais': df_metricas[df_metricas['Modelo'].str.contains('MLP|LSTM')]
}

for categoria, df_cat in categorias.items():
    if len(df_cat) > 0:
        print(f"\n{categoria}:")
        print(f"  Melhor AUC: {df_cat['AUC'].max():.4f} ({df_cat.loc[df_cat['AUC'].idxmax(), 'Modelo']})")
        print(f"  AUC médio: {df_cat['AUC'].mean():.4f}")
        print(f"  Número de modelos: {len(df_cat)}")

#%% Ranking final

print("\n" + "="*80)
print("RANKING FINAL")
print("="*80)

# Ordenar por AUC
df_ranking = df_metricas.sort_values('AUC', ascending=False)

print("Top 15 Modelos por AUC:")
print("="*50)
for i, (_, row) in enumerate(df_ranking.head(15).iterrows(), 1):
    print(f"{i:2d}. {row['Modelo']:25s} - AUC: {row['AUC']:.4f}, Gini: {row['Gini']:.4f}")

# Melhor modelo
melhor_modelo = df_ranking.iloc[0]
print(f"\nMELHOR MODELO:")
print(f"   {melhor_modelo['Modelo']}")
print(f"   AUC: {melhor_modelo['AUC']:.4f}")
print(f"   Gini: {melhor_modelo['Gini']:.4f}")
print(f"   Acurácia: {melhor_modelo['Acurácia']:.4f}")
print(f"   Precisão: {melhor_modelo['Precisão']:.4f}")
print(f"   Recall: {melhor_modelo['Recall']:.4f}")

#%% Análise de significância estatística

print("\n" + "="*80)
print("ANÁLISE DE SIGNIFICÂNCIA ESTATÍSTICA")
print("="*80)

# Teste t para comparar o melhor modelo com os outros
from scipy import stats

melhor_auc = melhor_modelo['AUC']
outros_aucs = df_ranking.iloc[1:]['AUC'].values

if len(outros_aucs) > 0:
    t_stat, p_value = stats.ttest_1samp(outros_aucs, melhor_auc)
    print(f"Teste t para o melhor modelo vs outros:")
    print(f"  Estatística t: {t_stat:.4f}")
    print(f"  P-valor: {p_value:.4f}")
    print(f"  Significativo (p < 0.05): {'Sim' if p_value < 0.05 else 'Não'}")

#%% Salvamento dos resultados

print("\n" + "="*80)
print("SALVAMENTO DOS RESULTADOS")
print("="*80)

# Salvar métricas
df_metricas.to_csv('metricas_todos_modelos.csv', index=False)

# Salvar ranking
df_ranking.to_csv('ranking_final_modelos.csv', index=False)

# Salvar resumo
resumo = {
    'Total_Modelos': len(df_metricas),
    'Melhor_Modelo': melhor_modelo['Modelo'],
    'Melhor_AUC': melhor_modelo['AUC'],
    'Melhor_Gini': melhor_modelo['Gini'],
    'AUC_Medio': df_metricas['AUC'].mean(),
    'AUC_Desvio': df_metricas['AUC'].std(),
    'AUC_Maximo': df_metricas['AUC'].max(),
    'AUC_Minimo': df_metricas['AUC'].min()
}

pd.DataFrame([resumo]).to_csv('resumo_final.csv', index=False)

print("Resultados salvos:")
print("- metricas_todos_modelos.csv")
print("- ranking_final_modelos.csv")
print("- resumo_final.csv")
print("- comparacao_completa_modelos.png")
print("- evolucao_complexidade.png")

#%% Resumo final

print("\n" + "="*80)
print("RESUMO FINAL - COMPARAÇÃO DE TODOS OS MODELOS")
print("="*80)

print(f"Total de modelos analisados: {len(df_metricas)}")
print(f"Melhor modelo: {melhor_modelo['Modelo']}")
print(f"Performance: {'Muito boa' if melhor_modelo['AUC'] > 0.8 else 'Boa' if melhor_modelo['AUC'] > 0.7 else 'Moderada'}")

print(f"\nDistribuição de performance:")
print(f"  AUC > 0.8: {len(df_metricas[df_metricas['AUC'] > 0.8])} modelos")
print(f"  AUC > 0.7: {len(df_metricas[df_metricas['AUC'] > 0.7])} modelos")
print(f"  AUC > 0.6: {len(df_metricas[df_metricas['AUC'] > 0.6])} modelos")

print(f"\nCategoria com melhor performance:")
melhor_categoria = max(categorias.items(), key=lambda x: x[1]['AUC'].max() if len(x[1]) > 0 else 0)
print(f"  {melhor_categoria[0]}: {melhor_categoria[1]['AUC'].max():.4f}")

print("\n" + "="*80)
print("COMPARAÇÃO FINAL CONCLUÍDA")
print("="*80)



