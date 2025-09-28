# -*- coding: utf-8 -*-
"""
UNIVERSIDADE DE SÃO PAULO
MBA DATA SCIENCE & ANALYTICS USP/ESALQ
PREDIÇÃO DE RECOLHIMENTO DE VEÍCULOS - ÁRVORES DE DECISÃO
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
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import (roc_auc_score, confusion_matrix, classification_report, 
                           roc_curve, auc, accuracy_score, precision_score, recall_score)
import warnings
warnings.filterwarnings('ignore')

#%% Configurações de visualização

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

#%% Funções auxiliares

def avalia_arvore(modelo, X, y, titulo="Avaliação da Árvore"):
    """Função para avaliação completa da árvore"""
    y_pred = modelo.predict(X)
    y_pred_proba = modelo.predict_proba(X)[:, 1]
    
    # Métricas
    auc = roc_auc_score(y, y_pred_proba)
    gini = 2 * auc - 1
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    
    print(f"\n{titulo}")
    print("="*50)
    print(f"AUC: {auc:.4f}")
    print(f"Gini: {gini:.4f}")
    print(f"Acurácia: {accuracy:.4f}")
    print(f"Precisão: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    # Matriz de confusão
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusão - {titulo}')
    plt.ylabel('Real')
    plt.xlabel('Predito')
    plt.savefig('arvore_plot.png', dpi=300, bbox_inches='tight'); plt.close()
    
    return {
        'auc': auc,
        'gini': gini,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

def plot_curva_roc(y_true, y_pred_proba, titulo="Curva ROC"):
    """Função para plotagem da curva ROC"""
    from sklearn.metrics import roc_curve, auc as auc_score
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc_score(fpr, tpr)
    gini = 2 * roc_auc - 1
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorchid', linewidth=3, 
             label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('1 - Especificidade', fontsize=14)
    plt.ylabel('Sensitividade', fontsize=14)
    plt.title(f'{titulo}\nGini = {gini:.4f}', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig('arvore_plot.png', dpi=300, bbox_inches='tight'); plt.close()
    
    return roc_auc, gini

def plot_importancia_features(modelo, feature_names, titulo="Importância das Features"):
    """Função para plotagem da importância das features"""
    importances = modelo.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title(titulo, fontsize=16)
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Importância', fontsize=12)
    plt.tight_layout()
    plt.savefig('arvore_plot.png', dpi=300, bbox_inches='tight'); plt.close()
    
    return importances

#%% Carregamento dos dados

print("="*80)
print("CARREGAMENTO DOS DADOS - PREDIÇÃO DE RECOLHIMENTO DE VEÍCULOS")
print("="*80)

# Carregar dataset mais recente
import os
from pathlib import Path

# Tentar diferentes caminhos possíveis
possible_paths = [
    Path("../../outputs"),
    Path("../outputs"), 
    Path("outputs"),
    Path("../../novo_tcc/outputs")
]

outputs_dir = None
for path in possible_paths:
    if path.exists() and list(path.glob("objective_dataset_clima_*.csv")):
        outputs_dir = path
        break

if not outputs_dir:
    raise FileNotFoundError("Nenhum dataset encontrado. Verifique se a pasta outputs existe.")

csv_files = list(outputs_dir.glob("objective_dataset_clima_*.csv"))
latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
df = pd.read_csv(latest_file)

print(f"Dataset carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")
print(f"Arquivo: {latest_file.name}")

#%% Preparação dos dados

print("\n" + "="*80)
print("PREPARAÇÃO DOS DADOS")
print("="*80)

# Converter data
if 'data_evento' in df.columns:
    df['data_evento'] = pd.to_datetime(df['data_evento'], errors='coerce')

# Preparar features
exclude = {
    'servico_id', 'manutID', 'manutencao_id', 'veiculo_code', 'veiculo_code_servico',
    'latitude', 'longitude', 'servico_tipo_id', 'recolhimento_evento', 'data_evento'
}

# Selecionar features numéricas
feature_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude]

# Encoding de classificacao_moto se existir
if 'classificacao_moto' in df.columns:
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df['classificacao_moto_encoded'] = le.fit_transform(df['classificacao_moto'].astype(str))
    if 'classificacao_moto_encoded' not in feature_cols:
        feature_cols.append('classificacao_moto_encoded')
    print(f"classificacao_moto encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Remover colunas duplicadas
feature_cols = list(set(feature_cols))

# Split temporal (últimos 20% como teste)
n_samples = len(df)
test_size = int(0.2 * n_samples)
train_idx = np.arange(n_samples - test_size)
test_idx = np.arange(n_samples - test_size, n_samples)

# Ordenar por data para split temporal
if 'data_evento' in df.columns:
    sort_idx = df['data_evento'].argsort()
    df = df.iloc[sort_idx]

X_train = df.iloc[train_idx][feature_cols].copy()
X_test = df.iloc[test_idx][feature_cols].copy()
y_train = df.iloc[train_idx]['recolhimento_evento'].copy()
y_test = df.iloc[test_idx]['recolhimento_evento'].copy()

# Tratar valores infinitos e NaN
X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

print(f"Split temporal: Treino={len(X_train)}, Teste={len(X_test)}")
print(f"Features selecionadas: {len(feature_cols)}")
print(f"Proporção de recolhimentos (treino): {y_train.mean():.3f}")
print(f"Proporção de recolhimentos (teste): {y_test.mean():.3f}")

#%% Análise exploratória

print("\n" + "="*80)
print("ANÁLISE EXPLORATÓRIA")
print("="*80)

# Estatísticas das principais features
print("Estatísticas das principais features:")
desc_cols = ['dias_desde_ultima_manutencao', 'total_7d', 'eletricas_7d', 'delta_km_desde_ultima_manutencao']
available_cols = [c for c in desc_cols if c in X_train.columns]
if available_cols:
    print(X_train[available_cols].describe())

# Distribuição da variável dependente
print(f"\nDistribuição da variável dependente:")
print(f"Treino - Não recolhido: {(y_train == 0).sum()} ({(y_train == 0).mean():.3f})")
print(f"Treino - Recolhido: {(y_train == 1).sum()} ({y_train.mean():.3f})")

#%% Árvore 1: Árvore Simples (sem poda)

print("\n" + "="*80)
print("ÁRVORE 1: ÁRVORE SIMPLES (SEM PODA)")
print("="*80)

# Criar árvore simples
arvore_simples = DecisionTreeClassifier(random_state=42)
arvore_simples.fit(X_train, y_train)

# Avaliar árvore simples
resultado_simples = avalia_arvore(arvore_simples, X_train, y_train, "Árvore Simples - Treino")
avalia_arvore(arvore_simples, X_test, y_test, "Árvore Simples - Teste")

# Curva ROC
roc_auc_simples, gini_simples = plot_curva_roc(y_test, arvore_simples.predict_proba(X_test)[:, 1], 
                                               "Curva ROC - Árvore Simples")

# Importância das features
importancias_simples = plot_importancia_features(arvore_simples, feature_cols, 
                                                "Importância das Features - Árvore Simples")

# Visualizar árvore (primeiros 3 níveis)
plt.figure(figsize=(20, 10))
plot_tree(arvore_simples, max_depth=3, feature_names=feature_cols, 
          class_names=['Não Recolhido', 'Recolhido'], filled=True, fontsize=10)
plt.title("Árvore de Decisão - Primeiros 3 Níveis", fontsize=16)
plt.tight_layout()
plt.savefig('arvore_simples_visualizacao.png', dpi=300, bbox_inches='tight')
plt.savefig('arvore_plot.png', dpi=300, bbox_inches='tight'); plt.close()

#%% Árvore 2: Árvore com Poda (Cost Complexity Pruning)

print("\n" + "="*80)
print("ÁRVORE 2: ÁRVORE COM PODA (COST COMPLEXITY PRUNING)")
print("="*80)

# Calcular path de poda
path = arvore_simples.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

print(f"Número de alfas únicos: {len(ccp_alphas)}")
print(f"Alfa mínimo: {ccp_alphas.min():.6f}")
print(f"Alfa máximo: {ccp_alphas.max():.6f}")

# Avaliar diferentes alfas
ginis = []
accuracies = []

for alpha in ccp_alphas:
    clf = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
    clf.fit(X_train, y_train)
    auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    gini = 2 * auc - 1
    accuracy = accuracy_score(y_test, clf.predict(X_test))
    ginis.append(gini)
    accuracies.append(accuracy)

# Plotar evolução do Gini e Acurácia
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(ccp_alphas, ginis, marker='o')
plt.xlabel('Alpha (CCP)', fontsize=12)
plt.ylabel('Gini', fontsize=12)
plt.title('Evolução do Gini vs Alpha', fontsize=14)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(ccp_alphas, accuracies, marker='o', color='orange')
plt.xlabel('Alpha (CCP)', fontsize=12)
plt.ylabel('Acurácia', fontsize=12)
plt.title('Evolução da Acurácia vs Alpha', fontsize=14)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('evolucao_poda.png', dpi=300, bbox_inches='tight')
plt.savefig('arvore_plot.png', dpi=300, bbox_inches='tight'); plt.close()

# Encontrar melhor alpha
melhor_alpha_idx = np.argmax(ginis)
melhor_alpha = ccp_alphas[melhor_alpha_idx]

print(f"Melhor alpha: {melhor_alpha:.6f}")
print(f"Melhor Gini: {ginis[melhor_alpha_idx]:.4f}")

# Criar árvore com melhor alpha
arvore_podada = DecisionTreeClassifier(ccp_alpha=melhor_alpha, random_state=42)
arvore_podada.fit(X_train, y_train)

# Avaliar árvore podada
resultado_podada = avalia_arvore(arvore_podada, X_train, y_train, "Árvore Podada - Treino")
avalia_arvore(arvore_podada, X_test, y_test, "Árvore Podada - Teste")

# Curva ROC
roc_auc_podada, gini_podada = plot_curva_roc(y_test, arvore_podada.predict_proba(X_test)[:, 1], 
                                             "Curva ROC - Árvore Podada")

# Importância das features
importancias_podada = plot_importancia_features(arvore_podada, feature_cols, 
                                               "Importância das Features - Árvore Podada")

# Visualizar árvore podada
plt.figure(figsize=(20, 10))
plot_tree(arvore_podada, max_depth=3, feature_names=feature_cols, 
          class_names=['Não Recolhido', 'Recolhido'], filled=True, fontsize=10)
plt.title("Árvore Podada - Primeiros 3 Níveis", fontsize=16)
plt.tight_layout()
plt.savefig('arvore_podada_visualizacao.png', dpi=300, bbox_inches='tight')
plt.savefig('arvore_plot.png', dpi=300, bbox_inches='tight'); plt.close()

#%% Árvore 3: Árvore com Grid Search

print("\n" + "="*80)
print("ÁRVORE 3: ÁRVORE COM GRID SEARCH")
print("="*80)

# Definir parâmetros para grid search
param_grid = {
    'max_depth': [3, 5, 7, 10, 15, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10],
    'ccp_alpha': [0, 0.001, 0.01, 0.1]
}

# Grid search com validação temporal
tscv = TimeSeriesSplit(n_splits=3)
grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=tscv,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

print("Executando Grid Search...")
grid_search.fit(X_train, y_train)

print(f"Melhores parâmetros: {grid_search.best_params_}")
print(f"Melhor score: {grid_search.best_score_:.4f}")

# Criar árvore com melhores parâmetros
arvore_otimizada = grid_search.best_estimator_

# Avaliar árvore otimizada
resultado_otimizada = avalia_arvore(arvore_otimizada, X_train, y_train, "Árvore Otimizada - Treino")
avalia_arvore(arvore_otimizada, X_test, y_test, "Árvore Otimizada - Teste")

# Curva ROC
roc_auc_otimizada, gini_otimizada = plot_curva_roc(y_test, arvore_otimizada.predict_proba(X_test)[:, 1], 
                                                   "Curva ROC - Árvore Otimizada")

# Importância das features
importancias_otimizada = plot_importancia_features(arvore_otimizada, feature_cols, 
                                                   "Importância das Features - Árvore Otimizada")

#%% Comparação das árvores

print("\n" + "="*80)
print("COMPARAÇÃO DAS ÁRVORES")
print("="*80)

# Comparação de performance
comparacao_arvores = pd.DataFrame({
    'Modelo': ['Simples', 'Podada', 'Otimizada'],
    'AUC': [roc_auc_simples, roc_auc_podada, roc_auc_otimizada],
    'Gini': [gini_simples, gini_podada, gini_otimizada],
    'Acurácia': [resultado_simples['accuracy'], resultado_podada['accuracy'], resultado_otimizada['accuracy']],
    'Precisão': [resultado_simples['precision'], resultado_podada['precision'], resultado_otimizada['precision']],
    'Recall': [resultado_simples['recall'], resultado_podada['recall'], resultado_otimizada['recall']]
})

print("Comparação de Performance das Árvores:")
print(comparacao_arvores)

# Gráfico de comparação
plt.figure(figsize=(15, 10))

# Curvas ROC comparativas
fpr_simples, tpr_simples, _ = roc_curve(y_test, arvore_simples.predict_proba(X_test)[:, 1])
fpr_podada, tpr_podada, _ = roc_curve(y_test, arvore_podada.predict_proba(X_test)[:, 1])
fpr_otimizada, tpr_otimizada, _ = roc_curve(y_test, arvore_otimizada.predict_proba(X_test)[:, 1])

plt.plot(fpr_simples, tpr_simples, color='blue', linewidth=3, 
         label=f'Simples (AUC = {roc_auc_simples:.4f})')
plt.plot(fpr_podada, tpr_podada, color='red', linewidth=3, 
         label=f'Podada (AUC = {roc_auc_podada:.4f})')
plt.plot(fpr_otimizada, tpr_otimizada, color='green', linewidth=3, 
         label=f'Otimizada (AUC = {roc_auc_otimizada:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.5)

plt.xlabel('1 - Especificidade', fontsize=14)
plt.ylabel('Sensitividade', fontsize=14)
plt.title('Comparação das Curvas ROC - Árvores de Decisão', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('comparacao_roc_arvores.png', dpi=300, bbox_inches='tight')
plt.savefig('arvore_plot.png', dpi=300, bbox_inches='tight'); plt.close()

# Gráfico de barras para métricas
plt.figure(figsize=(12, 8))
x = np.arange(len(comparacao_arvores))
width = 0.25

plt.bar(x - width, comparacao_arvores['AUC'], width, label='AUC', alpha=0.8)
plt.bar(x, comparacao_arvores['Gini'], width, label='Gini', alpha=0.8)
plt.bar(x + width, comparacao_arvores['Acurácia'], width, label='Acurácia', alpha=0.8)

plt.xlabel('Modelos', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Comparação de Métricas - Árvores de Decisão', fontsize=14)
plt.xticks(x, comparacao_arvores['Modelo'])
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('comparacao_metricas_arvores.png', dpi=300, bbox_inches='tight')
plt.savefig('arvore_plot.png', dpi=300, bbox_inches='tight'); plt.close()

#%% Análise de overfitting

print("\n" + "="*80)
print("ANÁLISE DE OVERFITTING")
print("="*80)

# Comparar performance treino vs teste
overfitting_analysis = pd.DataFrame({
    'Modelo': ['Simples', 'Podada', 'Otimizada'],
    'AUC_Treino': [roc_auc_score(y_train, arvore_simples.predict_proba(X_train)[:, 1]),
                   roc_auc_score(y_train, arvore_podada.predict_proba(X_train)[:, 1]),
                   roc_auc_score(y_train, arvore_otimizada.predict_proba(X_train)[:, 1])],
    'AUC_Teste': [roc_auc_simples, roc_auc_podada, roc_auc_otimizada]
})

overfitting_analysis['Diferenca'] = overfitting_analysis['AUC_Treino'] - overfitting_analysis['AUC_Teste']

print("Análise de Overfitting:")
print(overfitting_analysis)

# Gráfico de overfitting
plt.figure(figsize=(12, 8))
x = np.arange(len(overfitting_analysis))
width = 0.35

plt.bar(x - width/2, overfitting_analysis['AUC_Treino'], width, label='Treino', alpha=0.8)
plt.bar(x + width/2, overfitting_analysis['AUC_Teste'], width, label='Teste', alpha=0.8)

plt.xlabel('Modelos', fontsize=12)
plt.ylabel('AUC', fontsize=12)
plt.title('Análise de Overfitting - AUC Treino vs Teste', fontsize=14)
plt.xticks(x, overfitting_analysis['Modelo'])
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('overfitting_arvores.png', dpi=300, bbox_inches='tight')
plt.savefig('arvore_plot.png', dpi=300, bbox_inches='tight'); plt.close()

#%% Interpretabilidade da melhor árvore

print("\n" + "="*80)
print("INTERPRETABILIDADE DA MELHOR ÁRVORE")
print("="*80)

# Encontrar melhor árvore
melhor_arvore = arvore_otimizada
melhor_nome = "Otimizada"

if roc_auc_podada > roc_auc_otimizada:
    melhor_arvore = arvore_podada
    melhor_nome = "Podada"

if roc_auc_simples > max(roc_auc_podada, roc_auc_otimizada):
    melhor_arvore = arvore_simples
    melhor_nome = "Simples"

print(f"Melhor árvore: {melhor_nome}")

# Regras da árvore
print(f"\nRegras da árvore {melhor_nome}:")
tree_rules = export_text(melhor_arvore, feature_names=feature_cols, max_depth=3)
print(tree_rules)

# Top 10 features mais importantes
importancias = melhor_arvore.feature_importances_
indices = np.argsort(importancias)[::-1]

print(f"\nTop 10 Features Mais Importantes:")
for i in range(min(10, len(feature_cols))):
    idx = indices[i]
    print(f"{i+1:2d}. {feature_cols[idx]:30s} {importancias[idx]:.4f}")

#%% Salvamento dos resultados

print("\n" + "="*80)
print("SALVAMENTO DOS RESULTADOS")
print("="*80)

# Salvar modelos
import joblib
joblib.dump(arvore_simples, 'arvore_simples.pkl')
joblib.dump(arvore_podada, 'arvore_podada.pkl')
joblib.dump(arvore_otimizada, 'arvore_otimizada.pkl')

# Salvar predições
predicoes = pd.DataFrame({
    'y_true': y_test,
    'y_pred_simples': arvore_simples.predict(X_test),
    'y_pred_proba_simples': arvore_simples.predict_proba(X_test)[:, 1],
    'y_pred_podada': arvore_podada.predict(X_test),
    'y_pred_proba_podada': arvore_podada.predict_proba(X_test)[:, 1],
    'y_pred_otimizada': arvore_otimizada.predict(X_test),
    'y_pred_proba_otimizada': arvore_otimizada.predict_proba(X_test)[:, 1]
})

predicoes.to_csv('predicoes_arvores.csv', index=False)

# Salvar comparação
comparacao_arvores.to_csv('comparacao_arvores.csv', index=False)

print("Resultados salvos:")
print("- arvore_simples.pkl")
print("- arvore_podada.pkl")
print("- arvore_otimizada.pkl")
print("- predicoes_arvores.csv")
print("- comparacao_arvores.csv")
print("- Gráficos salvos como PNG")

#%% Resumo final

print("\n" + "="*80)
print("RESUMO FINAL - ÁRVORES DE DECISÃO")
print("="*80)

print(f"Árvore Simples:")
print(f"  - AUC: {roc_auc_simples:.4f}")
print(f"  - Gini: {gini_simples:.4f}")
print(f"  - Acurácia: {resultado_simples['accuracy']:.4f}")

print(f"\nÁrvore Podada:")
print(f"  - AUC: {roc_auc_podada:.4f}")
print(f"  - Gini: {gini_podada:.4f}")
print(f"  - Acurácia: {resultado_podada['accuracy']:.4f}")

print(f"\nÁrvore Otimizada:")
print(f"  - AUC: {roc_auc_otimizada:.4f}")
print(f"  - Gini: {gini_otimizada:.4f}")
print(f"  - Acurácia: {resultado_otimizada['accuracy']:.4f}")

print(f"\nMelhor árvore: {melhor_nome}")
print(f"Performance: {'Muito boa' if max(roc_auc_simples, roc_auc_podada, roc_auc_otimizada) > 0.8 else 'Boa' if max(roc_auc_simples, roc_auc_podada, roc_auc_otimizada) > 0.7 else 'Moderada'}")

print("\n" + "="*80)
print("ANÁLISE DE ÁRVORES DE DECISÃO CONCLUÍDA")
print("="*80)
