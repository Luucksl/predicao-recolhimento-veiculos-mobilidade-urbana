# -*- coding: utf-8 -*-
"""
UNIVERSIDADE DE SÃO PAULO
MBA DATA SCIENCE & ANALYTICS USP/ESALQ
PREDIÇÃO DE RECOLHIMENTO DE VEÍCULOS - ENSEMBLE METHODS
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
from sklearn.ensemble import (RandomForestClassifier, VotingClassifier, 
                            BaggingClassifier, AdaBoostClassifier, 
                            GradientBoostingClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import (roc_auc_score, confusion_matrix, classification_report, 
                           roc_curve, auc, accuracy_score, precision_score, recall_score)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Configuração de GPU para XGBoost
print("Configurando GPU para Ensemble...")
print(f"XGBoost version: {xgb.__version__}")

# Verificar se GPU está disponível para XGBoost
try:
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

def avalia_ensemble(modelo, X, y, titulo="Avaliação do Ensemble"):
    """Função para avaliação completa do ensemble"""
    y_pred = modelo.predict(X)
    
    # Verificar se o modelo tem predict_proba
    if hasattr(modelo, 'predict_proba'):
        y_pred_proba = modelo.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, y_pred_proba)
        gini = 2 * auc - 1
    else:
        # Para modelos sem predict_proba, usar apenas métricas baseadas em predições
        y_pred_proba = None
        auc = None
        gini = None
    
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    
    print(f"\n{titulo}")
    print("="*50)
    if auc is not None:
        print(f"AUC: {auc:.4f}")
        print(f"Gini: {gini:.4f}")
    else:
        print("AUC: N/A (modelo sem predict_proba)")
        print("Gini: N/A (modelo sem predict_proba)")
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
    plt.savefig('ensemble_plot.png', dpi=300, bbox_inches='tight'); plt.close()
    
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
    plt.savefig('ensemble_plot.png', dpi=300, bbox_inches='tight'); plt.close()
    
    return roc_auc, gini

def plot_comparacao_ensembles(resultados, titulo="Comparação de Ensembles"):
    """Função para plotagem da comparação de ensembles"""
    # Filtrar modelos com valores válidos
    modelos_validos = [modelo for modelo in resultados.keys() 
                      if resultados[modelo]['auc'] is not None and resultados[modelo]['gini'] is not None]
    aucs = [resultados[modelo]['auc'] for modelo in modelos_validos]
    ginis = [resultados[modelo]['gini'] for modelo in modelos_validos]
    
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: AUC
    plt.subplot(2, 2, 1)
    bars = plt.bar(modelos_validos, aucs, alpha=0.8)
    plt.ylabel('AUC', fontsize=12)
    plt.title('Comparação de AUC', fontsize=14)
    plt.xticks(rotation=45)
    for bar, auc in zip(bars, aucs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{auc:.3f}', ha='center', va='bottom')
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Gini
    plt.subplot(2, 2, 2)
    if ginis:  # Só plotar se houver valores de gini
        bars = plt.bar(modelos_validos, ginis, alpha=0.8, color='orange')
        plt.ylabel('Gini', fontsize=12)
        plt.title('Comparação de Gini', fontsize=14)
        plt.xticks(rotation=45)
        for bar, gini in zip(bars, ginis):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{gini:.3f}', ha='center', va='bottom')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Nenhum modelo com Gini disponível', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Comparação de Gini', fontsize=14)
    
    # Subplot 3: Curvas ROC
    plt.subplot(2, 2, 3)
    for modelo, resultado in resultados.items():
        if resultado['y_pred_proba'] is not None:
            fpr, tpr, _ = roc_curve(y_test, resultado['y_pred_proba'])
            auc_label = f'{resultado["auc"]:.3f}' if resultado["auc"] is not None else 'N/A'
            plt.plot(fpr, tpr, linewidth=2, label=f'{modelo} (AUC = {auc_label})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('1 - Especificidade', fontsize=12)
    plt.ylabel('Sensitividade', fontsize=12)
    plt.title('Curvas ROC', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Métricas múltiplas
    plt.subplot(2, 2, 4)
    accuracies = [resultados[modelo]['accuracy'] for modelo in modelos_validos]
    precisions = [resultados[modelo]['precision'] for modelo in modelos_validos]
    recalls = [resultados[modelo]['recall'] for modelo in modelos_validos]
    
    x = np.arange(len(modelos_validos))
    width = 0.25
    
    plt.bar(x - width, accuracies, width, label='Acurácia', alpha=0.8)
    plt.bar(x, precisions, width, label='Precisão', alpha=0.8)
    plt.bar(x + width, recalls, width, label='Recall', alpha=0.8)
    
    plt.xlabel('Modelos', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Métricas Múltiplas', fontsize=14)
    plt.xticks(x, modelos_validos, rotation=45)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{titulo.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.savefig('ensemble_plot.png', dpi=300, bbox_inches='tight'); plt.close()

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

#%% Criação dos modelos base

print("\n" + "="*80)
print("CRIAÇÃO DOS MODELOS BASE")
print("="*80)

# Modelos base para ensemble
modelos_base = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', verbosity=0, **gpu_params),
    'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
}

# Treinar modelos base
print("Treinando modelos base...")
for nome, modelo in modelos_base.items():
    print(f"Treinando {nome}...")
    modelo.fit(X_train, y_train)

# Avaliar modelos base
print("\nAvaliando modelos base...")
resultados_base = {}
for nome, modelo in modelos_base.items():
    resultado = avalia_ensemble(modelo, X_test, y_test, f"{nome} - Teste")
    resultados_base[nome] = resultado

#%% Ensemble 1: Voting Classifier (Hard Voting)

print("\n" + "="*80)
print("ENSEMBLE 1: VOTING CLASSIFIER (HARD VOTING)")
print("="*80)

# Voting Classifier com Hard Voting
voting_hard = VotingClassifier(
    estimators=list(modelos_base.items()),
    voting='hard'
)

voting_hard.fit(X_train, y_train)
resultado_hard = avalia_ensemble(voting_hard, X_test, y_test, "Voting Hard - Teste")

#%% Ensemble 2: Voting Classifier (Soft Voting)

print("\n" + "="*80)
print("ENSEMBLE 2: VOTING CLASSIFIER (SOFT VOTING)")
print("="*80)

# Voting Classifier com Soft Voting
voting_soft = VotingClassifier(
    estimators=list(modelos_base.items()),
    voting='soft'
)

voting_soft.fit(X_train, y_train)
resultado_soft = avalia_ensemble(voting_soft, X_test, y_test, "Voting Soft - Teste")

#%% Ensemble 3: Bagging Classifier

print("\n" + "="*80)
print("ENSEMBLE 3: BAGGING CLASSIFIER")
print("="*80)

# Bagging Classifier
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=100,
    random_state=42,
    n_jobs=1
)

bagging.fit(X_train, y_train)
resultado_bagging = avalia_ensemble(bagging, X_test, y_test, "Bagging - Teste")

#%% Ensemble 4: AdaBoost Classifier

print("\n" + "="*80)
print("ENSEMBLE 4: ADABOOST CLASSIFIER")
print("="*80)

# AdaBoost Classifier
adaboost = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=3, random_state=42),
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

adaboost.fit(X_train, y_train)
resultado_adaboost = avalia_ensemble(adaboost, X_test, y_test, "AdaBoost - Teste")

#%% Ensemble 5: Gradient Boosting Classifier

print("\n" + "="*80)
print("ENSEMBLE 5: GRADIENT BOOSTING CLASSIFIER")
print("="*80)

# Gradient Boosting Classifier
gradient_boost = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)

gradient_boost.fit(X_train, y_train)
resultado_gradient = avalia_ensemble(gradient_boost, X_test, y_test, "Gradient Boosting - Teste")

#%% Ensemble 6: Stacking Classifier

print("\n" + "="*80)
print("ENSEMBLE 6: STACKING CLASSIFIER")
print("="*80)

# Stacking Classifier
stacking = StackingClassifier(
    estimators=list(modelos_base.items()),
    final_estimator=LogisticRegression(random_state=42),
    cv=5,
    stack_method='predict_proba'
)

stacking.fit(X_train, y_train)
resultado_stacking = avalia_ensemble(stacking, X_test, y_test, "Stacking - Teste")

#%% Ensemble 7: Random Forest (como baseline)

print("\n" + "="*80)
print("ENSEMBLE 7: RANDOM FOREST (BASELINE)")
print("="*80)

# Random Forest como baseline
rf_baseline = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=1
)

rf_baseline.fit(X_train, y_train)
resultado_rf = avalia_ensemble(rf_baseline, X_test, y_test, "Random Forest - Teste")

#%% Comparação de todos os ensembles

print("\n" + "="*80)
print("COMPARAÇÃO DE TODOS OS ENSEMBLES")
print("="*80)

# Compilar resultados
resultados_ensemble = {
    'Voting Hard': resultado_hard,
    'Voting Soft': resultado_soft,
    'Bagging': resultado_bagging,
    'AdaBoost': resultado_adaboost,
    'Gradient Boosting': resultado_gradient,
    'Stacking': resultado_stacking,
    'Random Forest': resultado_rf
}

# Adicionar modelos base
resultados_ensemble.update(resultados_base)

# Plotar comparação
plot_comparacao_ensembles(resultados_ensemble, "Comparação de Todos os Modelos")

# Tabela de comparação
comparacao_ensemble = pd.DataFrame({
    'Modelo': list(resultados_ensemble.keys()),
    'AUC': [resultados_ensemble[modelo]['auc'] if resultados_ensemble[modelo]['auc'] is not None else 0 for modelo in resultados_ensemble.keys()],
    'Gini': [resultados_ensemble[modelo]['gini'] if resultados_ensemble[modelo]['gini'] is not None else 0 for modelo in resultados_ensemble.keys()],
    'Acurácia': [resultados_ensemble[modelo]['accuracy'] for modelo in resultados_ensemble.keys()],
    'Precisão': [resultados_ensemble[modelo]['precision'] for modelo in resultados_ensemble.keys()],
    'Recall': [resultados_ensemble[modelo]['recall'] for modelo in resultados_ensemble.keys()]
})

# Ordenar por AUC
comparacao_ensemble = comparacao_ensemble.sort_values('AUC', ascending=False)

print("Comparação de Performance (ordenado por AUC):")
print(comparacao_ensemble)

#%% Análise de validação cruzada temporal

print("\n" + "="*80)
print("ANÁLISE DE VALIDAÇÃO CRUZADA TEMPORAL")
print("="*80)

# Validação cruzada temporal para os melhores modelos
melhores_modelos = {
    'Voting Soft': voting_soft,
    'Stacking': stacking,
    'Random Forest': rf_baseline,
    'XGBoost': modelos_base['XGBoost']
}

tscv = TimeSeriesSplit(n_splits=5)
cv_resultados = {}

for nome, modelo in melhores_modelos.items():
    print(f"Executando validação cruzada para {nome}...")
    scores = cross_val_score(modelo, X_train, y_train, cv=tscv, scoring='roc_auc')
    cv_resultados[nome] = {
        'scores': scores,
        'mean': scores.mean(),
        'std': scores.std()
    }
    print(f"  AUC médio: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

# Plotar resultados da validação cruzada
plt.figure(figsize=(12, 8))
modelos_cv = list(cv_resultados.keys())
means = [cv_resultados[modelo]['mean'] for modelo in modelos_cv]
stds = [cv_resultados[modelo]['std'] for modelo in modelos_cv]

plt.bar(modelos_cv, means, yerr=stds, capsize=5, alpha=0.8)
plt.ylabel('AUC Score', fontsize=12)
plt.title('Validação Cruzada Temporal - AUC Score', fontsize=14)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('validacao_cruzada_temporal.png', dpi=300, bbox_inches='tight')
plt.savefig('ensemble_plot.png', dpi=300, bbox_inches='tight'); plt.close()

#%% Análise de diversidade dos ensembles

print("\n" + "="*80)
print("ANÁLISE DE DIVERSIDADE DOS ENSEMBLES")
print("="*80)

# Calcular correlação entre predições dos modelos
predicoes_modelos = {}
for nome, resultado in resultados_ensemble.items():
    predicoes_modelos[nome] = resultado['y_pred_proba']

df_predicoes = pd.DataFrame(predicoes_modelos)

# Matriz de correlação
correlacao = df_predicoes.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlacao, annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.3f')
plt.title('Matriz de Correlação - Predições dos Modelos', fontsize=16)
plt.tight_layout()
plt.savefig('correlacao_modelos.png', dpi=300, bbox_inches='tight')
plt.savefig('ensemble_plot.png', dpi=300, bbox_inches='tight'); plt.close()

# Análise de diversidade
print("Análise de Diversidade:")
print("Correlações mais altas (modelos mais similares):")
correlacao_upper = correlacao.where(np.triu(np.ones(correlacao.shape), k=1).astype(bool))
max_corr = correlacao_upper.max().max()
print(f"Correlação máxima: {max_corr:.3f}")

#%% Interpretabilidade do melhor ensemble

print("\n" + "="*80)
print("INTERPRETABILIDADE DO MELHOR ENSEMBLE")
print("="*80)

# Encontrar melhor modelo
melhor_modelo_nome = comparacao_ensemble.iloc[0]['Modelo']
melhor_modelo_auc = comparacao_ensemble.iloc[0]['AUC']

print(f"Melhor modelo: {melhor_modelo_nome} (AUC: {melhor_modelo_auc:.4f})")

# Análise de importância das features (se disponível)
if hasattr(resultados_ensemble[melhor_modelo_nome], 'feature_importances_'):
    importancias = resultados_ensemble[melhor_modelo_nome].feature_importances_
    indices = np.argsort(importancias)[::-1]
    
    print(f"\nTop 10 Features Mais Importantes ({melhor_modelo_nome}):")
    for i in range(min(10, len(feature_cols))):
        idx = indices[i]
        print(f"{i+1:2d}. {feature_cols[idx]:30s} {importancias[idx]:.4f}")
    
    # Gráfico de importância
    plt.figure(figsize=(12, 8))
    top_features = min(15, len(feature_cols))
    plt.barh(range(top_features), importancias[indices[:top_features]])
    plt.yticks(range(top_features), [feature_cols[i] for i in indices[:top_features]])
    plt.xlabel('Importância', fontsize=12)
    plt.title(f'Top 15 Features Mais Importantes - {melhor_modelo_nome}', fontsize=16)
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('importancia_features_ensemble.png', dpi=300, bbox_inches='tight')
    plt.savefig('ensemble_plot.png', dpi=300, bbox_inches='tight'); plt.close()

#%% Salvamento dos resultados

print("\n" + "="*80)
print("SALVAMENTO DOS RESULTADOS")
print("="*80)

# Salvar modelos
import joblib
for nome, modelo in modelos_base.items():
    joblib.dump(modelo, f'modelo_base_{nome.lower()}.pkl')

# joblib.dump(voting_hard, 'voting_hard.pkl')
# joblib.dump(voting_soft, 'voting_soft.pkl')
# joblib.dump(bagging, 'bagging.pkl')
# joblib.dump(adaboost, 'adaboost.pkl')
# joblib.dump(gradient_boost, 'gradient_boost.pkl')
# joblib.dump(stacking, 'stacking.pkl')
# joblib.dump(rf_baseline, 'rf_baseline.pkl')
print("Modelos ensemble não salvos para evitar problemas de serialização")

# Salvar predições
predicoes_ensemble = pd.DataFrame({
    'y_true': y_test,
    **{nome: resultado['y_pred_proba'] for nome, resultado in resultados_ensemble.items()}
})

predicoes_ensemble.to_csv('predicoes_ensemble.csv', index=False)

# Salvar comparação
comparacao_ensemble.to_csv('comparacao_ensemble.csv', index=False)

print("Resultados salvos:")
print("- Modelos base: modelo_base_*.pkl")
print("- Ensembles: voting_*.pkl, bagging.pkl, adaboost.pkl, etc.")
print("- predicoes_ensemble.csv")
print("- comparacao_ensemble.csv")
print("- Gráficos salvos como PNG")

#%% Resumo final

print("\n" + "="*80)
print("RESUMO FINAL - ENSEMBLE METHODS")
print("="*80)

print("Top 5 Modelos por AUC:")
top_5 = comparacao_ensemble.head(5)
for i, (_, row) in enumerate(top_5.iterrows(), 1):
    print(f"{i}. {row['Modelo']:20s} - AUC: {row['AUC']:.4f}, Gini: {row['Gini']:.4f}")

print(f"\nMelhor modelo: {melhor_modelo_nome}")
print(f"Performance: {'Muito boa' if melhor_modelo_auc > 0.8 else 'Boa' if melhor_modelo_auc > 0.7 else 'Moderada'}")

print(f"\nValidação cruzada temporal:")
for nome, cv in cv_resultados.items():
    print(f"  {nome:15s}: {cv['mean']:.4f} (+/- {cv['std'] * 2:.4f})")

print("\n" + "="*80)
print("ANÁLISE DE ENSEMBLE METHODS CONCLUÍDA")
print("="*80)
