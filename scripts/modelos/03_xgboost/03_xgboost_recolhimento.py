# -*- coding: utf-8 -*-
"""
UNIVERSIDADE DE SÃO PAULO
MBA DATA SCIENCE & ANALYTICS USP/ESALQ
PREDIÇÃO DE RECOLHIMENTO DE VEÍCULOS - XGBOOST
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
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import (roc_auc_score, confusion_matrix, classification_report, 
                           roc_curve, auc, accuracy_score, precision_score, recall_score)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Configuração de GPU para XGBoost
print("Configurando XGBoost com GPU...")
print(f"XGBoost version: {xgb.__version__}")

# Verificar se GPU está disponível para XGBoost
try:
    # Testar se GPU está disponível
    import subprocess
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        print("OK - GPU NVIDIA detectada - XGBoost usará GPU")
        gpu_params = {'tree_method': 'gpu_hist', 'gpu_id': 0}
    else:
        print("AVISO - GPU NVIDIA não detectada - XGBoost usará CPU")
        gpu_params = {}
except:
    print("ERRO - Erro ao detectar GPU - XGBoost usará CPU")
    gpu_params = {}

#%% Configurações de visualização

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

#%% Funções auxiliares

def avalia_xgboost(modelo, X, y, titulo="Avaliação do XGBoost"):
    """Função para avaliação completa do XGBoost"""
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
    plt.savefig('xgboost_plot.png', dpi=300, bbox_inches='tight'); plt.close()
    
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
    plt.savefig('xgboost_plot.png', dpi=300, bbox_inches='tight'); plt.close()
    
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
    plt.savefig('xgboost_plot.png', dpi=300, bbox_inches='tight'); plt.close()
    
    return importances

def plot_learning_curve(modelo, X_train, y_train, X_val, y_val, titulo="Learning Curve"):
    """Função para plotagem da curva de aprendizado"""
    # Usar eval_set para obter histórico de treinamento
    eval_set = [(X_train, y_train), (X_val, y_val)]
    eval_names = ['train', 'validation']
    
    # Treinar modelo com early stopping
    modelo.fit(X_train, y_train, 
               eval_set=eval_set,
               verbose=False)
    
    # Obter histórico de treinamento
    results = modelo.evals_result()
    
    # Plotar curva de aprendizado
    plt.figure(figsize=(12, 8))
    for i, name in enumerate(eval_names):
        plt.plot(results['validation_0' if i == 0 else 'validation_1']['logloss'], 
                label=f'{name.capitalize()}', linewidth=2)
    
    plt.xlabel('Número de Árvores', fontsize=12)
    plt.ylabel('Log Loss', fontsize=12)
    plt.title(f'{titulo} - XGBoost', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig('xgboost_plot.png', dpi=300, bbox_inches='tight'); plt.close()

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

#%% XGBoost 1: XGBoost Simples (parâmetros padrão)

print("\n" + "="*80)
print("XGBOOST 1: XGBOOST SIMPLES (PARÂMETROS PADRÃO)")
print("="*80)

# Criar XGBoost simples
xgb_simples = xgb.XGBClassifier(
    random_state=42,
    eval_metric='logloss',
    verbosity=0,
    **gpu_params
)

xgb_simples.fit(X_train, y_train)

# Avaliar XGBoost simples
resultado_simples = avalia_xgboost(xgb_simples, X_train, y_train, "XGBoost Simples - Treino")
avalia_xgboost(xgb_simples, X_test, y_test, "XGBoost Simples - Teste")

# Curva ROC
roc_auc_simples, gini_simples = plot_curva_roc(y_test, xgb_simples.predict_proba(X_test)[:, 1], 
                                               "Curva ROC - XGBoost Simples")

# Importância das features
importancias_simples = plot_importancia_features(xgb_simples, feature_cols, 
                                                "Importância das Features - XGBoost Simples")

#%% XGBoost 2: XGBoost com Early Stopping

print("\n" + "="*80)
print("XGBOOST 2: XGBOOST COM EARLY STOPPING")
print("="*80)

# Split adicional para validação
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# XGBoost com early stopping
xgb_early = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    eval_metric='logloss',
    verbosity=0,
    **gpu_params
)

# Treinar com early stopping
xgb_early.fit(
    X_train_split, y_train_split,
    eval_set=[(X_val_split, y_val_split)],
    verbose=False
)

try:
    print(f"Número de árvores treinadas: {xgb_early.best_iteration}")
except AttributeError:
    print(f"Número de árvores treinadas: {xgb_early.n_estimators}")

# Avaliar XGBoost com early stopping
resultado_early = avalia_xgboost(xgb_early, X_train, y_train, "XGBoost Early Stopping - Treino")
avalia_xgboost(xgb_early, X_test, y_test, "XGBoost Early Stopping - Teste")

# Curva ROC
roc_auc_early, gini_early = plot_curva_roc(y_test, xgb_early.predict_proba(X_test)[:, 1], 
                                           "Curva ROC - XGBoost Early Stopping")

# Importância das features
importancias_early = plot_importancia_features(xgb_early, feature_cols, 
                                              "Importância das Features - XGBoost Early Stopping")

# Curva de aprendizado
plot_learning_curve(xgb_early, X_train_split, y_train_split, X_val_split, y_val_split,
                   "Curva de Aprendizado - XGBoost Early Stopping")

#%% XGBoost 3: XGBoost com Randomized Search

print("\n" + "="*80)
print("XGBOOST 3: XGBOOST COM RANDOMIZED SEARCH")
print("="*80)

# Definir parâmetros para randomized search
param_dist = {
    'n_estimators': [100, 200, 300, 500, 1000],
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    'max_depth': [3, 4, 5, 6, 7, 8, 10],
    'min_child_weight': [1, 3, 5, 7],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'reg_alpha': [0, 0.1, 0.5, 1, 10],
    'reg_lambda': [0, 0.1, 0.5, 1, 10]
}

# Randomized search com validação temporal
tscv = TimeSeriesSplit(n_splits=3)
random_search = RandomizedSearchCV(
    xgb.XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0, **gpu_params),
    param_dist,
    n_iter=50,  # Número de iterações
    cv=tscv,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

print("Executando Randomized Search...")
random_search.fit(X_train, y_train)

print(f"Melhores parâmetros: {random_search.best_params_}")
print(f"Melhor score: {random_search.best_score_:.4f}")

# Criar XGBoost com melhores parâmetros
xgb_otimizado = random_search.best_estimator_

# Avaliar XGBoost otimizado
resultado_otimizado = avalia_xgboost(xgb_otimizado, X_train, y_train, "XGBoost Otimizado - Treino")
avalia_xgboost(xgb_otimizado, X_test, y_test, "XGBoost Otimizado - Teste")

# Curva ROC
roc_auc_otimizado, gini_otimizado = plot_curva_roc(y_test, xgb_otimizado.predict_proba(X_test)[:, 1], 
                                                   "Curva ROC - XGBoost Otimizado")

# Importância das features
importancias_otimizado = plot_importancia_features(xgb_otimizado, feature_cols, 
                                                   "Importância das Features - XGBoost Otimizado")

#%% XGBoost 4: XGBoost com Parâmetros Específicos para Dados Temporais

print("\n" + "="*80)
print("XGBOOST 4: XGBOOST COM PARÂMETROS ESPECÍFICOS PARA DADOS TEMPORAIS")
print("="*80)

# XGBoost com parâmetros específicos para dados temporais
xgb_temporal = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    eval_metric='logloss',
    verbosity=0,
    **gpu_params
)

xgb_temporal.fit(X_train, y_train)

# Avaliar XGBoost temporal
resultado_temporal = avalia_xgboost(xgb_temporal, X_train, y_train, "XGBoost Temporal - Treino")
avalia_xgboost(xgb_temporal, X_test, y_test, "XGBoost Temporal - Teste")

# Curva ROC
roc_auc_temporal, gini_temporal = plot_curva_roc(y_test, xgb_temporal.predict_proba(X_test)[:, 1], 
                                                 "Curva ROC - XGBoost Temporal")

# Importância das features
importancias_temporal = plot_importancia_features(xgb_temporal, feature_cols, 
                                                 "Importância das Features - XGBoost Temporal")

#%% Comparação dos modelos XGBoost

print("\n" + "="*80)
print("COMPARAÇÃO DOS MODELOS XGBOOST")
print("="*80)

# Comparação de performance
comparacao_xgb = pd.DataFrame({
    'Modelo': ['Simples', 'Early Stopping', 'Otimizado', 'Temporal'],
    'AUC': [roc_auc_simples, roc_auc_early, roc_auc_otimizado, roc_auc_temporal],
    'Gini': [gini_simples, gini_early, gini_otimizado, gini_temporal],
    'Acurácia': [resultado_simples['accuracy'], resultado_early['accuracy'], 
                 resultado_otimizado['accuracy'], resultado_temporal['accuracy']],
    'Precisão': [resultado_simples['precision'], resultado_early['precision'], 
                 resultado_otimizado['precision'], resultado_temporal['precision']],
    'Recall': [resultado_simples['recall'], resultado_early['recall'], 
               resultado_otimizado['recall'], resultado_temporal['recall']]
})

print("Comparação de Performance dos Modelos XGBoost:")
print(comparacao_xgb)

# Gráfico de comparação
plt.figure(figsize=(15, 10))

# Curvas ROC comparativas
fpr_simples, tpr_simples, _ = roc_curve(y_test, xgb_simples.predict_proba(X_test)[:, 1])
fpr_early, tpr_early, _ = roc_curve(y_test, xgb_early.predict_proba(X_test)[:, 1])
fpr_otimizado, tpr_otimizado, _ = roc_curve(y_test, xgb_otimizado.predict_proba(X_test)[:, 1])
fpr_temporal, tpr_temporal, _ = roc_curve(y_test, xgb_temporal.predict_proba(X_test)[:, 1])

plt.plot(fpr_simples, tpr_simples, color='blue', linewidth=3, 
         label=f'Simples (AUC = {roc_auc_simples:.4f})')
plt.plot(fpr_early, tpr_early, color='red', linewidth=3, 
         label=f'Early Stopping (AUC = {roc_auc_early:.4f})')
plt.plot(fpr_otimizado, tpr_otimizado, color='green', linewidth=3, 
         label=f'Otimizado (AUC = {roc_auc_otimizado:.4f})')
plt.plot(fpr_temporal, tpr_temporal, color='orange', linewidth=3, 
         label=f'Temporal (AUC = {roc_auc_temporal:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.5)

plt.xlabel('1 - Especificidade', fontsize=14)
plt.ylabel('Sensitividade', fontsize=14)
plt.title('Comparação das Curvas ROC - XGBoost', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('comparacao_roc_xgboost.png', dpi=300, bbox_inches='tight')
plt.savefig('xgboost_plot.png', dpi=300, bbox_inches='tight'); plt.close()

# Gráfico de barras para métricas
plt.figure(figsize=(12, 8))
x = np.arange(len(comparacao_xgb))
width = 0.25

plt.bar(x - width, comparacao_xgb['AUC'], width, label='AUC', alpha=0.8)
plt.bar(x, comparacao_xgb['Gini'], width, label='Gini', alpha=0.8)
plt.bar(x + width, comparacao_xgb['Acurácia'], width, label='Acurácia', alpha=0.8)

plt.xlabel('Modelos', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Comparação de Métricas - XGBoost', fontsize=14)
plt.xticks(x, comparacao_xgb['Modelo'])
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('comparacao_metricas_xgboost.png', dpi=300, bbox_inches='tight')
plt.savefig('xgboost_plot.png', dpi=300, bbox_inches='tight'); plt.close()

#%% Análise de overfitting

print("\n" + "="*80)
print("ANÁLISE DE OVERFITTING")
print("="*80)

# Comparar performance treino vs teste
overfitting_analysis = pd.DataFrame({
    'Modelo': ['Simples', 'Early Stopping', 'Otimizado', 'Temporal'],
    'AUC_Treino': [roc_auc_score(y_train, xgb_simples.predict_proba(X_train)[:, 1]),
                   roc_auc_score(y_train, xgb_early.predict_proba(X_train)[:, 1]),
                   roc_auc_score(y_train, xgb_otimizado.predict_proba(X_train)[:, 1]),
                   roc_auc_score(y_train, xgb_temporal.predict_proba(X_train)[:, 1])],
    'AUC_Teste': [roc_auc_simples, roc_auc_early, roc_auc_otimizado, roc_auc_temporal]
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
plt.savefig('overfitting_xgboost.png', dpi=300, bbox_inches='tight')
plt.savefig('xgboost_plot.png', dpi=300, bbox_inches='tight'); plt.close()

#%% Análise de importância das features (melhor modelo)

print("\n" + "="*80)
print("ANÁLISE DE IMPORTÂNCIA DAS FEATURES - MELHOR MODELO")
print("="*80)

# Encontrar melhor modelo
modelos_xgb = {
    'Simples': xgb_simples,
    'Early Stopping': xgb_early,
    'Otimizado': xgb_otimizado,
    'Temporal': xgb_temporal
}

aucs = [roc_auc_simples, roc_auc_early, roc_auc_otimizado, roc_auc_temporal]
melhor_idx = np.argmax(aucs)
melhor_nome = list(modelos_xgb.keys())[melhor_idx]
melhor_modelo = list(modelos_xgb.values())[melhor_idx]

print(f"Melhor modelo XGBoost: {melhor_nome}")

# Top 15 features mais importantes
importancias = melhor_modelo.feature_importances_
indices = np.argsort(importancias)[::-1]

print(f"\nTop 15 Features Mais Importantes ({melhor_nome}):")
for i in range(min(15, len(feature_cols))):
    idx = indices[i]
    print(f"{i+1:2d}. {feature_cols[idx]:30s} {importancias[idx]:.4f}")

# Gráfico de importância das features
plt.figure(figsize=(15, 10))
top_features = min(15, len(feature_cols))
plt.barh(range(top_features), importancias[indices[:top_features]])
plt.yticks(range(top_features), [feature_cols[i] for i in indices[:top_features]])
plt.xlabel('Importância', fontsize=12)
plt.title(f'Top 15 Features Mais Importantes - {melhor_nome}', fontsize=16)
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('importancia_features_xgboost.png', dpi=300, bbox_inches='tight')
plt.savefig('xgboost_plot.png', dpi=300, bbox_inches='tight'); plt.close()

#%% Salvamento dos resultados

print("\n" + "="*80)
print("SALVAMENTO DOS RESULTADOS")
print("="*80)

# Salvar modelos
import joblib
joblib.dump(xgb_simples, 'xgboost_simples.pkl')
joblib.dump(xgb_early, 'xgboost_early.pkl')
joblib.dump(xgb_otimizado, 'xgboost_otimizado.pkl')
joblib.dump(xgb_temporal, 'xgboost_temporal.pkl')

# Salvar predições
predicoes = pd.DataFrame({
    'y_true': y_test,
    'y_pred_simples': xgb_simples.predict(X_test),
    'y_pred_proba_simples': xgb_simples.predict_proba(X_test)[:, 1],
    'y_pred_early': xgb_early.predict(X_test),
    'y_pred_proba_early': xgb_early.predict_proba(X_test)[:, 1],
    'y_pred_otimizado': xgb_otimizado.predict(X_test),
    'y_pred_proba_otimizado': xgb_otimizado.predict_proba(X_test)[:, 1],
    'y_pred_temporal': xgb_temporal.predict(X_test),
    'y_pred_proba_temporal': xgb_temporal.predict_proba(X_test)[:, 1]
})

predicoes.to_csv('predicoes_xgboost.csv', index=False)

# Salvar comparação
comparacao_xgb.to_csv('comparacao_xgboost.csv', index=False)

print("Resultados salvos:")
print("- xgboost_simples.pkl")
print("- xgboost_early.pkl")
print("- xgboost_otimizado.pkl")
print("- xgboost_temporal.pkl")
print("- predicoes_xgboost.csv")
print("- comparacao_xgboost.csv")
print("- Gráficos salvos como PNG")

#%% Resumo final

print("\n" + "="*80)
print("RESUMO FINAL - XGBOOST")
print("="*80)

print(f"XGBoost Simples:")
print(f"  - AUC: {roc_auc_simples:.4f}")
print(f"  - Gini: {gini_simples:.4f}")
print(f"  - Acurácia: {resultado_simples['accuracy']:.4f}")

print(f"\nXGBoost Early Stopping:")
print(f"  - AUC: {roc_auc_early:.4f}")
print(f"  - Gini: {gini_early:.4f}")
print(f"  - Acurácia: {resultado_early['accuracy']:.4f}")

print(f"\nXGBoost Otimizado:")
print(f"  - AUC: {roc_auc_otimizado:.4f}")
print(f"  - Gini: {gini_otimizado:.4f}")
print(f"  - Acurácia: {resultado_otimizado['accuracy']:.4f}")

print(f"\nXGBoost Temporal:")
print(f"  - AUC: {roc_auc_temporal:.4f}")
print(f"  - Gini: {gini_temporal:.4f}")
print(f"  - Acurácia: {resultado_temporal['accuracy']:.4f}")

print(f"\nMelhor modelo: {melhor_nome}")
print(f"Performance: {'Muito boa' if max(aucs) > 0.8 else 'Boa' if max(aucs) > 0.7 else 'Moderada'}")

print("\n" + "="*80)
print("ANÁLISE DE XGBOOST CONCLUÍDA")
print("="*80)
