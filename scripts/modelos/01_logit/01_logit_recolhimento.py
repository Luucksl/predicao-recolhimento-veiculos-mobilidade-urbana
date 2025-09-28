# -*- coding: utf-8 -*-
"""
UNIVERSIDADE DE SÃO PAULO
MBA DATA SCIENCE & ANALYTICS USP/ESALQ
PREDIÇÃO DE RECOLHIMENTO DE VEÍCULOS - REGRESSÃO LOGÍSTICA
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
from scipy.interpolate import UnivariateSpline
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statstests.process import stepwise
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

#%% Configurações de visualização

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

#%% Funções auxiliares

def prob(z):
    """Função para cálculo da probabilidade sigmoide"""
    return 1 / (1 + np.exp(-z))

def matriz_confusao(predicts, observado, cutoff):
    """Função para construção da matriz de confusão"""
    from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay, recall_score
    
    values = predicts.values
    predicao_binaria = []
    
    for item in values:
        if item < cutoff:
            predicao_binaria.append(0)
        else:
            predicao_binaria.append(1)
    
    cm = confusion_matrix(predicao_binaria, observado)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.xlabel('True')
    plt.ylabel('Classified')
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.title(f'Matriz de Confusão - Cutoff = {cutoff}')
    plt.savefig(f'logit_{i}.png', dpi=300, bbox_inches='tight'); plt.close()
    
    sensitividade = recall_score(observado, predicao_binaria, pos_label=1)
    especificidade = recall_score(observado, predicao_binaria, pos_label=0)
    acuracia = accuracy_score(observado, predicao_binaria)
    
    indicadores = pd.DataFrame({
        'Sensitividade': [sensitividade],
        'Especificidade': [especificidade],
        'Acurácia': [acuracia]
    })
    return indicadores

def espec_sens(observado, predicts):
    """Função para análise de sensitividade e especificidade"""
    from sklearn.metrics import recall_score
    
    values = predicts.values
    cutoffs = np.arange(0, 1.01, 0.01)
    
    lista_sensitividade = []
    lista_especificidade = []
    
    for cutoff in cutoffs:
        predicao_binaria = []
        
        for item in values:
            if item >= cutoff:
                predicao_binaria.append(1)
            else:
                predicao_binaria.append(0)
        
        sensitividade = recall_score(observado, predicao_binaria, pos_label=1)
        especificidadee = recall_score(observado, predicao_binaria, pos_label=0)
        
        lista_sensitividade.append(sensitividade)
        lista_especificidade.append(especificidadee)
    
    resultado = pd.DataFrame({
        'cutoffs': cutoffs,
        'sensitividade': lista_sensitividade,
        'especificidade': lista_especificidade
    })
    return resultado

def plot_curva_sigmoide(df, x_var, y_var, title="Curva Sigmoide"):
    """Função para plotagem da curva sigmoide"""
    plt.figure(figsize=(15, 10))
    
    # Scatter plot dos dados
    sns.scatterplot(x=df[x_var][df[y_var] == 0], y=df[y_var][df[y_var] == 0],
                   color='springgreen', alpha=0.7, s=250, label='Não Recolhido')
    sns.scatterplot(x=df[x_var][df[y_var] == 1], y=df[y_var][df[y_var] == 1],
                   color='magenta', alpha=0.7, s=250, label='Recolhido')
    
    # Curva sigmoide
    sns.regplot(x=df[x_var], y=df[y_var], logistic=True, ci=None, scatter=False,
               line_kws={'color': 'indigo', 'linewidth': 7})
    
    plt.axhline(y=0.5, color='grey', linestyle=':')
    plt.xlabel(x_var, fontsize=20)
    plt.ylabel('Probabilidade de Recolhimento', fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=20, loc='center right')
    plt.title(title, fontsize=22)
    plt.savefig(f'logit_{i}.png', dpi=300, bbox_inches='tight'); plt.close()

def plot_curva_roc(y_true, y_pred_proba, title="Curva ROC"):
    """Função para plotagem da curva ROC"""
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    gini = (roc_auc - 0.5) / 0.5
    
    plt.figure(figsize=(15, 10))
    plt.plot(fpr, tpr, marker='o', color='darkorchid', markersize=10, linewidth=3)
    plt.plot(fpr, fpr, color='gray', linestyle='dashed')
    plt.title(f'{title}\nÁrea abaixo da curva: {roc_auc:.4f} | Coeficiente de GINI: {gini:.4f}', 
              fontsize=22)
    plt.xlabel('1 - Especificidade', fontsize=20)
    plt.ylabel('Sensitividade', fontsize=20)
    plt.xticks(np.arange(0, 1.1, 0.2), fontsize=14)
    plt.yticks(np.arange(0, 1.1, 0.2), fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig('curva_roc_logit.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return roc_auc, gini

#%% Carregamento dos dados

print("="*80)
print("CARREGAMENTO DOS DADOS - PREDIÇÃO DE RECOLHIMENTO DE VEÍCULOS")
print("="*80)

# Carregar dataset mais recente
import os
from pathlib import Path

outputs_dir = Path("../../outputs")
csv_files = list(outputs_dir.glob("objective_dataset_clima_*.csv"))
if not csv_files:
    raise FileNotFoundError("Nenhum dataset encontrado em outputs/")

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

# Preparar features para regressão logística
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

df_train = df.iloc[train_idx].copy()
df_test = df.iloc[test_idx].copy()

print(f"Split temporal: Treino={len(df_train)}, Teste={len(df_test)}")
print(f"Features selecionadas: {len(feature_cols)}")

#%% Análise exploratória

print("\n" + "="*80)
print("ANÁLISE EXPLORATÓRIA")
print("="*80)

# Estatísticas da variável dependente
print("Distribuição da variável dependente:")
print(df_train['recolhimento_evento'].value_counts().sort_index())
print(f"Proporção de recolhimentos: {df_train['recolhimento_evento'].mean():.3f}")

# Estatísticas descritivas
print("\nEstatísticas descritivas das principais features:")
desc_cols = ['dias_desde_ultima_manutencao', 'total_7d', 'eletricas_7d', 'delta_km_desde_ultima_manutencao']
available_cols = [c for c in desc_cols if c in df_train.columns]
if available_cols:
    print(df_train[available_cols].describe())

#%% Construção da curva sigmoide teórica

print("\n" + "="*80)
print("CURVA SIGMOIDE TEÓRICA")
print("="*80)

# Plotagem da curva sigmoide teórica
logitos = []
probs = []

for i in np.arange(-5, 6):
    logitos.append(i)
    probs.append(prob(i))

df_sigmoide = pd.DataFrame({'logito': logitos, 'probs': probs})

# Interpolação spline (smooth probability line)
spline = UnivariateSpline(df_sigmoide['logito'], df_sigmoide['probs'], s=0)
logitos_smooth = np.linspace(df_sigmoide['logito'].min(), df_sigmoide['logito'].max(), 500)
probs_smooth = spline(logitos_smooth)

plt.figure(figsize=(15, 10))
plt.plot(logitos_smooth, probs_smooth, color='royalblue', linestyle='--', label='Prob. Evento')
plt.scatter(df_sigmoide['logito'], df_sigmoide['probs'], color='royalblue', marker='o', s=250)
plt.axhline(y=df_sigmoide.probs.mean(), color='grey', linestyle=':', xmax=0.5)
plt.axvline(x=0, color='grey', linestyle=':', ymax=0.5)
plt.xlabel("Logito Z", fontsize=20)
plt.ylabel("Probabilidade", fontsize=20)
plt.xticks(np.arange(-5, 6), fontsize=14)
plt.yticks(np.arange(0, 1.1, 0.1), fontsize=14)
plt.legend(fontsize=18, loc='center right')
plt.title("Curva Sigmoide Teórica - Função Logística", fontsize=22)
plt.grid(True, alpha=0.3)
plt.savefig('curva_sigmoide_teorica.png', dpi=300, bbox_inches='tight')
plt.savefig(f'logit_{i}.png', dpi=300, bbox_inches='tight'); plt.close()

#%% Modelo 1: Regressão Logística Simples

print("\n" + "="*80)
print("MODELO 1: REGRESSÃO LOGÍSTICA SIMPLES")
print("="*80)

# Selecionar features mais importantes para modelo simples
features_simples = ['dias_desde_ultima_manutencao', 'total_7d', 'eletricas_7d']
features_simples = [f for f in features_simples if f in df_train.columns]

if len(features_simples) < 2:
    # Usar as primeiras features numéricas disponíveis
    features_simples = feature_cols[:3]

print(f"Features para modelo simples: {features_simples}")

# Preparar dados para statsmodels
X_train_sm = df_train[features_simples].copy()
X_test_sm = df_test[features_simples].copy()
y_train = df_train['recolhimento_evento'].copy()
y_test = df_test['recolhimento_evento'].copy()

# Tratar valores infinitos e NaN
X_train_sm = X_train_sm.replace([np.inf, -np.inf], np.nan)
X_test_sm = X_test_sm.replace([np.inf, -np.inf], np.nan)
X_train_sm = X_train_sm.fillna(0)
X_test_sm = X_test_sm.fillna(0)

# Adicionar constante
X_train_sm = sm.add_constant(X_train_sm)
X_test_sm = sm.add_constant(X_test_sm)

# Estimação do modelo logístico
formula = 'recolhimento_evento ~ ' + ' + '.join(features_simples)
print(f"Fórmula do modelo: {formula}")

modelo_logit = sm.Logit(y_train, X_train_sm).fit()

# Resumo do modelo
print("\nResumo do Modelo Logístico:")
print(modelo_logit.summary())

#%% Predições e avaliação do modelo simples

print("\n" + "="*80)
print("PREDIÇÕES E AVALIAÇÃO - MODELO SIMPLES")
print("="*80)

# Predições
y_pred_proba_train = modelo_logit.predict(X_train_sm)
y_pred_proba_test = modelo_logit.predict(X_test_sm)

# Adicionar predições ao dataframe
df_train['phat_simples'] = y_pred_proba_train
df_test['phat_simples'] = y_pred_proba_test

print("Predições calculadas para treino e teste")

# Matriz de confusão - Cutoff 0.5
print("\nMatriz de Confusão - Cutoff 0.5:")
matriz_confusao(predicts=df_test['phat_simples'], 
                observado=df_test['recolhimento_evento'], 
                cutoff=0.5)

# Curva ROC
roc_auc, gini = plot_curva_roc(df_test['recolhimento_evento'], 
                               df_test['phat_simples'],
                               "Curva ROC - Modelo Simples")

print(f"\nMétricas do Modelo Simples:")
print(f"AUC: {roc_auc:.4f}")
print(f"Gini: {gini:.4f}")

#%% Análise de sensitividade e especificidade

print("\n" + "="*80)
print("ANÁLISE DE SENSITIVIDADE E ESPECIFICIDADE")
print("="*80)

# Análise de cutoff
dados_plotagem = espec_sens(observado=df_test['recolhimento_evento'],
                           predicts=df_test['phat_simples'])

plt.figure(figsize=(15, 10))
plt.plot(dados_plotagem.cutoffs, dados_plotagem.sensitividade, marker='o',
         color='indigo', markersize=8, label='Sensitividade')
plt.plot(dados_plotagem.cutoffs, dados_plotagem.especificidade, marker='o',
         color='limegreen', markersize=8, label='Especificidade')
plt.xlabel('Cutoff', fontsize=20)
plt.ylabel('Sensitividade / Especificidade', fontsize=20)
plt.xticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.yticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.legend(fontsize=20)
plt.title("Sensitividade vs Especificidade - Modelo Simples", fontsize=22)
plt.grid(True, alpha=0.3)
plt.savefig('sens_espec_simples.png', dpi=300, bbox_inches='tight')
plt.savefig(f'logit_{i}.png', dpi=300, bbox_inches='tight'); plt.close()

#%% Modelo 2: Regressão Logística com Stepwise

print("\n" + "="*80)
print("MODELO 2: REGRESSÃO LOGÍSTICA COM STEPWISE")
print("="*80)

# Preparar dados com todas as features
X_train_full = df_train[feature_cols].copy()
X_test_full = df_test[feature_cols].copy()

# Tratar valores infinitos e NaN
X_train_full = X_train_full.replace([np.inf, -np.inf], np.nan)
X_test_full = X_test_full.replace([np.inf, -np.inf], np.nan)
X_train_full = X_train_full.fillna(0)
X_test_full = X_test_full.fillna(0)

# Adicionar constante
X_train_full = sm.add_constant(X_train_full)
X_test_full = sm.add_constant(X_test_full)

# Usar apenas features principais para evitar multicolinearidade
features_principais = [
    'dias_desde_ultima_manutencao', 'eletricas_7d', 'eletricas_30d',
    'delta_km_desde_ultima_manutencao', 'classificacao_moto_encoded'
]

# Filtrar apenas features que existem no dataset
features_principais = [f for f in features_principais if f in feature_cols]
print(f"Features principais selecionadas: {features_principais}")

# Preparar dados com features principais
X_train_principais = df_train[features_principais].copy()
X_test_principais = df_test[features_principais].copy()

# Tratar valores infinitos e NaN
X_train_principais = X_train_principais.replace([np.inf, -np.inf], np.nan)
X_test_principais = X_test_principais.replace([np.inf, -np.inf], np.nan)
X_train_principais = X_train_principais.fillna(0)
X_test_principais = X_test_principais.fillna(0)

# Adicionar constante
X_train_principais = sm.add_constant(X_train_principais)
X_test_principais = sm.add_constant(X_test_principais)

# Modelo com features principais
formula_principais = 'recolhimento_evento ~ ' + ' + '.join(features_principais)
print(f"Fórmula do modelo com features principais: {formula_principais}")

modelo_completo = sm.Logit(y_train, X_train_principais).fit()

# Procedimento Stepwise
print("\nAplicando procedimento Stepwise...")
try:
    step_modelo = stepwise(modelo_completo, pvalue_limit=0.05)
    print("Stepwise concluído com sucesso!")
except:
    print("Erro no Stepwise. Usando modelo completo.")
    step_modelo = modelo_completo

# Resumo do modelo stepwise
print("\nResumo do Modelo Stepwise:")
print(step_modelo.summary())

#%% Predições e avaliação do modelo stepwise

print("\n" + "="*80)
print("PREDIÇÕES E AVALIAÇÃO - MODELO STEPWISE")
print("="*80)

# Predições
y_pred_proba_train_step = step_modelo.predict(X_train_principais)
y_pred_proba_test_step = step_modelo.predict(X_test_principais)

# Adicionar predições ao dataframe
df_train['phat_stepwise'] = y_pred_proba_train_step
df_test['phat_stepwise'] = y_pred_proba_test_step

print("Predições calculadas para modelo stepwise")

# Matriz de confusão - Cutoff 0.5
print("\nMatriz de Confusão - Cutoff 0.5:")
matriz_confusao(predicts=df_test['phat_stepwise'], 
                observado=df_test['recolhimento_evento'], 
                cutoff=0.5)

# Curva ROC
roc_auc_step, gini_step = plot_curva_roc(df_test['recolhimento_evento'], 
                                         df_test['phat_stepwise'],
                                         "Curva ROC - Modelo Stepwise")

print(f"\nMétricas do Modelo Stepwise:")
print(f"AUC: {roc_auc_step:.4f}")
print(f"Gini: {gini_step:.4f}")

#%% Comparação dos modelos

print("\n" + "="*80)
print("COMPARAÇÃO DOS MODELOS")
print("="*80)

# Comparação de performance
comparacao = pd.DataFrame({
    'Modelo': ['Simples', 'Stepwise'],
    'AUC': [roc_auc, roc_auc_step],
    'Gini': [gini, gini_step]
})

print("Comparação de Performance:")
print(comparacao)

# Gráfico de comparação
plt.figure(figsize=(15, 10))

# Curvas ROC comparativas
from sklearn.metrics import roc_curve, auc

fpr_simples, tpr_simples, _ = roc_curve(df_test['recolhimento_evento'], df_test['phat_simples'])
fpr_step, tpr_step, _ = roc_curve(df_test['recolhimento_evento'], df_test['phat_stepwise'])

plt.plot(fpr_simples, tpr_simples, color='blue', linewidth=3, 
         label=f'Modelo Simples (AUC = {roc_auc:.4f})')
plt.plot(fpr_step, tpr_step, color='red', linewidth=3, 
         label=f'Modelo Stepwise (AUC = {roc_auc_step:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.5)

plt.xlabel('1 - Especificidade', fontsize=20)
plt.ylabel('Sensitividade', fontsize=20)
plt.title('Comparação das Curvas ROC - Regressão Logística', fontsize=22)
plt.legend(fontsize=16)
plt.grid(True, alpha=0.3)
plt.savefig('comparacao_roc_logit.png', dpi=300, bbox_inches='tight')
plt.savefig(f'logit_{i}.png', dpi=300, bbox_inches='tight'); plt.close()

#%% Análise de significância estatística

print("\n" + "="*80)
print("ANÁLISE DE SIGNIFICÂNCIA ESTATÍSTICA")
print("="*80)

# Teste de significância global
print("Teste de Significância Global:")
print(f"Log-Likelihood: {step_modelo.llf:.4f}")
print(f"Log-Likelihood Null: {step_modelo.llnull:.4f}")
print(f"LR Statistic: {step_modelo.llr:.4f}")
print(f"LR p-value: {step_modelo.llr_pvalue:.4f}")

# Pseudo R²
print(f"\nPseudo R² (McFadden): {step_modelo.prsquared:.4f}")

#%% Interpretação dos coeficientes

print("\n" + "="*80)
print("INTERPRETAÇÃO DOS COEFICIENTES")
print("="*80)

# Coeficientes do modelo stepwise
coef_df = pd.DataFrame({
    'Variável': step_modelo.params.index,
    'Coeficiente': step_modelo.params.values,
    'P-valor': step_modelo.pvalues.values,
    'Odds Ratio': np.exp(step_modelo.params.values)
})

coef_df = coef_df[coef_df['Variável'] != 'const'].sort_values('P-valor')
print("Coeficientes do Modelo Stepwise (ordenados por p-valor):")
print(coef_df)

# Gráfico de coeficientes
plt.figure(figsize=(12, 8))
coef_plot = coef_df.head(10)  # Top 10 variáveis mais significativas
plt.barh(range(len(coef_plot)), coef_plot['Coeficiente'])
plt.yticks(range(len(coef_plot)), coef_plot['Variável'])
plt.xlabel('Coeficiente', fontsize=14)
plt.title('Coeficientes do Modelo Logístico - Top 10 Variáveis', fontsize=16)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('coeficientes_logit.png', dpi=300, bbox_inches='tight')
plt.savefig(f'logit_{i}.png', dpi=300, bbox_inches='tight'); plt.close()

#%% Salvamento dos resultados

print("\n" + "="*80)
print("SALVAMENTO DOS RESULTADOS")
print("="*80)

# Salvar modelos
import joblib
joblib.dump(modelo_logit, 'modelo_logit_simples.pkl')
joblib.dump(step_modelo, 'modelo_logit_stepwise.pkl')

# Salvar predições
df_test[['recolhimento_evento', 'phat_simples', 'phat_stepwise']].to_csv('predicoes_logit.csv', index=False)

# Salvar comparação
comparacao.to_csv('comparacao_modelos_logit.csv', index=False)

print("Resultados salvos:")
print("- modelo_logit_simples.pkl")
print("- modelo_logit_stepwise.pkl")
print("- predicoes_logit.csv")
print("- comparacao_modelos_logit.csv")
print("- Gráficos salvos como PNG")

#%% Resumo final

print("\n" + "="*80)
print("RESUMO FINAL - REGRESSÃO LOGÍSTICA")
print("="*80)

print(f"Modelo Simples:")
print(f"  - AUC: {roc_auc:.4f}")
print(f"  - Gini: {gini:.4f}")
print(f"  - Features: {len(features_simples)}")

print(f"\nModelo Stepwise:")
print(f"  - AUC: {roc_auc_step:.4f}")
print(f"  - Gini: {gini_step:.4f}")
print(f"  - Features: {len(step_modelo.params)-1}")

print(f"\nMelhor modelo: {'Stepwise' if roc_auc_step > roc_auc else 'Simples'}")
print(f"Performance: {'Muito boa' if max(roc_auc, roc_auc_step) > 0.8 else 'Boa' if max(roc_auc, roc_auc_step) > 0.7 else 'Moderada'}")

print("\n" + "="*80)
print("ANÁLISE DE REGRESSÃO LOGÍSTICA CONCLUÍDA")
print("="*80)
