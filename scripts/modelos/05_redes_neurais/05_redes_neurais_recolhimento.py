# -*- coding: utf-8 -*-
"""
UNIVERSIDADE DE SÃO PAULO
MBA DATA SCIENCE & ANALYTICS USP/ESALQ
PREDIÇÃO DE RECOLHIMENTO DE VEÍCULOS - REDES NEURAIS
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
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import (roc_auc_score, confusion_matrix, classification_report, 
                           roc_curve, auc, accuracy_score, precision_score, recall_score)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

# Configuração de GPU
print("Configurando GPU...")
print(f"TensorFlow version: {tf.__version__}")
print(f"GPUs disponíveis: {len(tf.config.list_physical_devices('GPU'))}")

# Configurar GPU se disponível
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Habilitar crescimento de memória para evitar alocação de toda a GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"OK - GPU configurada: {gpus[0].name}")
        print("GPU Memory Growth habilitado")
    except RuntimeError as e:
        print(f"ERRO - Erro ao configurar GPU: {e}")
else:
    print("AVISO - Nenhuma GPU detectada, usando CPU")

# Configurar para usar GPU se disponível
tf.config.set_visible_devices(gpus if gpus else [], 'GPU')

#%% Configurações de visualização

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

#%% Funções auxiliares

def avalia_rede_neural(modelo, X, y, titulo="Avaliação da Rede Neural"):
    """Função para avaliação completa da rede neural"""
    if hasattr(modelo, 'predict_proba'):
        y_pred_proba = modelo.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
    else:
        y_pred_proba = modelo.predict(X).flatten()
        y_pred = (y_pred_proba >= 0.5).astype(int)
    
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
    salvar_figura_segura(plt.gcf(), 'neural_plot.png')
    
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
    salvar_figura_segura(plt.gcf(), 'neural_plot.png')
    
    return roc_auc, gini

def plot_learning_curve(history, titulo="Learning Curve"):
    """Função para plotagem da curva de aprendizado"""
    plt.figure(figsize=(15, 5))
    
    # Subplot 1: Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Treino', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validação', linewidth=2)
    plt.title(f'{titulo} - Loss', fontsize=14)
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Treino', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validação', linewidth=2)
    plt.title(f'{titulo} - Accuracy', fontsize=14)
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{titulo.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    salvar_figura_segura(plt.gcf(), 'neural_plot.png')

def create_mlp_model(input_dim, hidden_layers, learning_rate=0.001):
    """Função para criar modelo MLP com TensorFlow"""
    model = keras.Sequential()
    
    # Camada de entrada
    model.add(layers.Dense(hidden_layers[0], activation='relu', input_shape=(input_dim,)))
    model.add(layers.Dropout(0.2))
    
    # Camadas ocultas
    for units in hidden_layers[1:]:
        model.add(layers.Dense(units, activation='relu'))
        model.add(layers.Dropout(0.2))
    
    # Camada de saída
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # Compilar modelo
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_lstm_model(input_dim, sequence_length, hidden_units=50):
    """Função para criar modelo LSTM com TensorFlow"""
    model = keras.Sequential()
    
    # Camada LSTM
    model.add(layers.LSTM(hidden_units, input_shape=(sequence_length, input_dim), return_sequences=True))
    model.add(layers.Dropout(0.2))
    
    # Segunda camada LSTM
    model.add(layers.LSTM(hidden_units, return_sequences=False))
    model.add(layers.Dropout(0.2))
    
    # Camadas densas
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(25, activation='relu'))
    
    # Camada de saída
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # Compilar modelo
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

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

# Normalização dos dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Dados normalizados com StandardScaler")

#%% Rede Neural 1: MLP Simples (Scikit-learn)

print("\n" + "="*80)
print("REDE NEURAL 1: MLP SIMPLES (SCIKIT-LEARN)")
print("="*80)

# MLP simples
mlp_simples = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    learning_rate='adaptive',
    max_iter=500,
    random_state=42
)

mlp_simples.fit(X_train_scaled, y_train)
resultado_mlp_simples = avalia_rede_neural(mlp_simples, X_test_scaled, y_test, "MLP Simples - Teste")

# Curva ROC
roc_auc_mlp_simples, gini_mlp_simples = plot_curva_roc(y_test, mlp_simples.predict_proba(X_test_scaled)[:, 1], 
                                                       "Curva ROC - MLP Simples")

#%% Rede Neural 2: MLP com TensorFlow

print("\n" + "="*80)
print("REDE NEURAL 2: MLP COM TENSORFLOW")
print("="*80)

# Criar modelo MLP com TensorFlow
mlp_tf = create_mlp_model(
    input_dim=len(feature_cols),
    hidden_layers=[128, 64, 32],
    learning_rate=0.001
)

print("Arquitetura do modelo MLP:")
mlp_tf.summary()

# Split adicional para validação
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Treinar modelo
history_mlp = mlp_tf.fit(
    X_train_split, y_train_split,
    validation_data=(X_val_split, y_val_split),
    epochs=100,
    batch_size=32,
    verbose=0
)

# Plotar curva de aprendizado
plot_learning_curve(history_mlp, "MLP TensorFlow")

# Avaliar modelo
y_pred_proba_mlp_tf = mlp_tf.predict(X_test_scaled).flatten()
resultado_mlp_tf = avalia_rede_neural(mlp_tf, X_test_scaled, y_test, "MLP TensorFlow - Teste")

# Curva ROC
roc_auc_mlp_tf, gini_mlp_tf = plot_curva_roc(y_test, y_pred_proba_mlp_tf, 
                                             "Curva ROC - MLP TensorFlow")

#%% Rede Neural 3: MLP com Grid Search

print("\n" + "="*80)
print("REDE NEURAL 3: MLP COM GRID SEARCH")
print("="*80)

# Grid search para MLP
param_grid_mlp = {
    'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 50, 25)],
    'activation': ['relu', 'tanh'],
    'learning_rate': ['constant', 'adaptive'],
    'alpha': [0.0001, 0.001, 0.01]
}

grid_search_mlp = GridSearchCV(
    MLPClassifier(max_iter=500, random_state=42),
    param_grid_mlp,
    cv=3,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

print("Executando Grid Search para MLP...")
grid_search_mlp.fit(X_train_scaled, y_train)

print(f"Melhores parâmetros: {grid_search_mlp.best_params_}")
print(f"Melhor score: {grid_search_mlp.best_score_:.4f}")

# Avaliar melhor MLP
mlp_otimizado = grid_search_mlp.best_estimator_
resultado_mlp_otimizado = avalia_rede_neural(mlp_otimizado, X_test_scaled, y_test, "MLP Otimizado - Teste")

# Curva ROC
roc_auc_mlp_otimizado, gini_mlp_otimizado = plot_curva_roc(y_test, mlp_otimizado.predict_proba(X_test_scaled)[:, 1], 
                                                           "Curva ROC - MLP Otimizado")

#%% Rede Neural 4: LSTM (para dados temporais)

print("\n" + "="*80)
print("REDE NEURAL 4: LSTM (PARA DADOS TEMPORAIS)")
print("="*80)

# Preparar dados para LSTM (sequências temporais)
def create_sequences(data, sequence_length=7):
    """Criar sequências temporais para LSTM"""
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequences.append(data[i:i + sequence_length])
    return np.array(sequences)

# Criar sequências temporais
sequence_length = 7
X_train_seq = create_sequences(X_train_scaled, sequence_length)
X_test_seq = create_sequences(X_test_scaled, sequence_length)

# Ajustar y para sequências
y_train_seq = y_train[sequence_length-1:]
y_test_seq = y_test[sequence_length-1:]

print(f"Sequências criadas: Treino={X_train_seq.shape}, Teste={X_test_seq.shape}")

# Criar modelo LSTM
lstm_model = create_lstm_model(
    input_dim=len(feature_cols),
    sequence_length=sequence_length,
    hidden_units=50
)

print("Arquitetura do modelo LSTM:")
lstm_model.summary()

# Split adicional para validação
X_train_lstm_split, X_val_lstm_split, y_train_lstm_split, y_val_lstm_split = train_test_split(
    X_train_seq, y_train_seq, test_size=0.2, random_state=42, stratify=y_train_seq
)

# Treinar modelo LSTM
history_lstm = lstm_model.fit(
    X_train_lstm_split, y_train_lstm_split,
    validation_data=(X_val_lstm_split, y_val_lstm_split),
    epochs=100,
    batch_size=32,
    verbose=0
)

# Plotar curva de aprendizado
plot_learning_curve(history_lstm, "LSTM")

# Avaliar modelo LSTM
y_pred_proba_lstm = lstm_model.predict(X_test_seq).flatten()
resultado_lstm = avalia_rede_neural(lstm_model, X_test_seq, y_test_seq, "LSTM - Teste")

# Curva ROC
roc_auc_lstm, gini_lstm = plot_curva_roc(y_test_seq, y_pred_proba_lstm, 
                                         "Curva ROC - LSTM")

#%% Rede Neural 5: MLP Profunda

print("\n" + "="*80)
print("REDE NEURAL 5: MLP PROFUNDA")
print("="*80)

# MLP profunda
mlp_profunda = create_mlp_model(
    input_dim=len(feature_cols),
    hidden_layers=[256, 128, 64, 32, 16],
    learning_rate=0.0005
)

print("Arquitetura do modelo MLP Profunda:")
mlp_profunda.summary()

# Treinar modelo
history_profunda = mlp_profunda.fit(
    X_train_split, y_train_split,
    validation_data=(X_val_split, y_val_split),
    epochs=150,
    batch_size=64,
    verbose=0
)

# Plotar curva de aprendizado
plot_learning_curve(history_profunda, "MLP Profunda")

# Avaliar modelo
y_pred_proba_profunda = mlp_profunda.predict(X_test_scaled).flatten()
resultado_profunda = avalia_rede_neural(mlp_profunda, X_test_scaled, y_test, "MLP Profunda - Teste")

# Curva ROC
roc_auc_profunda, gini_profunda = plot_curva_roc(y_test, y_pred_proba_profunda, 
                                                 "Curva ROC - MLP Profunda")

#%% Comparação das redes neurais

print("\n" + "="*80)
print("COMPARAÇÃO DAS REDES NEURAIS")
print("="*80)

# Compilar resultados
resultados_rn = {
    'MLP Simples': resultado_mlp_simples,
    'MLP TensorFlow': resultado_mlp_tf,
    'MLP Otimizado': resultado_mlp_otimizado,
    'LSTM': resultado_lstm,
    'MLP Profunda': resultado_profunda
}

# Tabela de comparação
comparacao_rn = pd.DataFrame({
    'Modelo': list(resultados_rn.keys()),
    'AUC': [resultados_rn[modelo]['auc'] for modelo in resultados_rn.keys()],
    'Gini': [resultados_rn[modelo]['gini'] for modelo in resultados_rn.keys()],
    'Acurácia': [resultados_rn[modelo]['accuracy'] for modelo in resultados_rn.keys()],
    'Precisão': [resultados_rn[modelo]['precision'] for modelo in resultados_rn.keys()],
    'Recall': [resultados_rn[modelo]['recall'] for modelo in resultados_rn.keys()]
})

# Ordenar por AUC
comparacao_rn = comparacao_rn.sort_values('AUC', ascending=False)

print("Comparação de Performance das Redes Neurais:")
print(comparacao_rn)

# Gráfico de comparação
plt.figure(figsize=(15, 10))

# Subplot 1: AUC
plt.subplot(2, 2, 1)
bars = plt.bar(comparacao_rn['Modelo'], comparacao_rn['AUC'], alpha=0.8)
plt.ylabel('AUC', fontsize=12)
plt.title('Comparação de AUC', fontsize=14)
plt.xticks(rotation=45)
for bar, auc in zip(bars, comparacao_rn['AUC']):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
            f'{auc:.3f}', ha='center', va='bottom')
plt.grid(True, alpha=0.3)

# Subplot 2: Gini
plt.subplot(2, 2, 2)
bars = plt.bar(comparacao_rn['Modelo'], comparacao_rn['Gini'], alpha=0.8, color='orange')
plt.ylabel('Gini', fontsize=12)
plt.title('Comparação de Gini', fontsize=14)
plt.xticks(rotation=45)
for bar, gini in zip(bars, comparacao_rn['Gini']):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
            f'{gini:.3f}', ha='center', va='bottom')
plt.grid(True, alpha=0.3)

# Subplot 3: Curvas ROC
plt.subplot(2, 2, 3)
for modelo, resultado in resultados_rn.items():
    # Verificar se os tamanhos são compatíveis
    y_true = y_test[:len(resultado['y_pred_proba'])]
    fpr, tpr, _ = roc_curve(y_true, resultado['y_pred_proba'])
    plt.plot(fpr, tpr, linewidth=2, label=f'{modelo} (AUC = {resultado["auc"]:.3f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.5)
plt.xlabel('1 - Especificidade', fontsize=12)
plt.ylabel('Sensitividade', fontsize=12)
plt.title('Curvas ROC', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Subplot 4: Métricas múltiplas
plt.subplot(2, 2, 4)
x = np.arange(len(comparacao_rn))
width = 0.25

plt.bar(x - width, comparacao_rn['Acurácia'], width, label='Acurácia', alpha=0.8)
plt.bar(x, comparacao_rn['Precisão'], width, label='Precisão', alpha=0.8)
plt.bar(x + width, comparacao_rn['Recall'], width, label='Recall', alpha=0.8)

plt.xlabel('Modelos', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Métricas Múltiplas', fontsize=14)
plt.xticks(x, comparacao_rn['Modelo'], rotation=45)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
salvar_figura_segura(plt.gcf(), 'comparacao_redes_neurais.png')
plt.savefig('neural_plot.png', dpi=300, bbox_inches='tight'); plt.close()

#%% Análise de overfitting

print("\n" + "="*80)
print("ANÁLISE DE OVERFITTING")
print("="*80)

# Comparar performance treino vs teste
overfitting_analysis = pd.DataFrame({
    'Modelo': ['MLP Simples', 'MLP TensorFlow', 'MLP Otimizado', 'LSTM', 'MLP Profunda'],
    'AUC_Treino': [
        roc_auc_score(y_train, mlp_simples.predict_proba(X_train_scaled)[:, 1]),
        roc_auc_score(y_train, mlp_tf.predict(X_train_scaled).flatten()),
        roc_auc_score(y_train, mlp_otimizado.predict_proba(X_train_scaled)[:, 1]),
        roc_auc_score(y_train_seq, lstm_model.predict(X_train_seq).flatten()),
        roc_auc_score(y_train, mlp_profunda.predict(X_train_scaled).flatten())
    ],
    'AUC_Teste': [
        resultado_mlp_simples['auc'],
        resultado_mlp_tf['auc'],
        resultado_mlp_otimizado['auc'],
        resultado_lstm['auc'],
        resultado_profunda['auc']
    ]
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
plt.xticks(x, overfitting_analysis['Modelo'], rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
salvar_figura_segura(plt.gcf(), 'overfitting_redes_neurais.png')
plt.savefig('neural_plot.png', dpi=300, bbox_inches='tight'); plt.close()

#%% Salvamento dos resultados

print("\n" + "="*80)
print("SALVAMENTO DOS RESULTADOS")
print("="*80)

# Salvar modelos
import joblib
joblib.dump(mlp_simples, 'mlp_simples.pkl')
joblib.dump(mlp_otimizado, 'mlp_otimizado.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Salvar modelos TensorFlow
mlp_tf.save('mlp_tensorflow.h5')
lstm_model.save('lstm_model.h5')
mlp_profunda.save('mlp_profunda.h5')

# Salvar predições (ajustando tamanhos para compatibilidade)
predicoes_rn = pd.DataFrame({
    'y_true': y_test,
    'y_pred_mlp_simples': mlp_simples.predict(X_test_scaled),
    'y_pred_proba_mlp_simples': mlp_simples.predict_proba(X_test_scaled)[:, 1],
    'y_pred_proba_mlp_tf': mlp_tf.predict(X_test_scaled).flatten(),
    'y_pred_mlp_otimizado': mlp_otimizado.predict(X_test_scaled),
    'y_pred_proba_mlp_otimizado': mlp_otimizado.predict_proba(X_test_scaled)[:, 1],
    'y_pred_proba_lstm': np.concatenate([np.zeros(len(y_test) - len(y_pred_proba_lstm)), y_pred_proba_lstm]),
    'y_pred_proba_mlp_profunda': mlp_profunda.predict(X_test_scaled).flatten()
})

predicoes_rn.to_csv('predicoes_redes_neurais.csv', index=False)

# Salvar comparação
comparacao_rn.to_csv('comparacao_redes_neurais.csv', index=False)

print("Resultados salvos:")
print("- mlp_simples.pkl")
print("- mlp_otimizado.pkl")
print("- scaler.pkl")
print("- mlp_tensorflow.h5")
print("- lstm_model.h5")
print("- mlp_profunda.h5")
print("- predicoes_redes_neurais.csv")
print("- comparacao_redes_neurais.csv")
print("- Gráficos salvos como PNG")

#%% Resumo final

print("\n" + "="*80)
print("RESUMO FINAL - REDES NEURAIS")
print("="*80)

print("Top 3 Redes Neurais por AUC:")
top_3 = comparacao_rn.head(3)
for i, (_, row) in enumerate(top_3.iterrows(), 1):
    print(f"{i}. {row['Modelo']:20s} - AUC: {row['AUC']:.4f}, Gini: {row['Gini']:.4f}")

melhor_rn = comparacao_rn.iloc[0]
print(f"\nMelhor rede neural: {melhor_rn['Modelo']}")
print(f"Performance: {'Muito boa' if melhor_rn['AUC'] > 0.8 else 'Boa' if melhor_rn['AUC'] > 0.7 else 'Moderada'}")

print(f"\nAnálise de Overfitting:")
for _, row in overfitting_analysis.iterrows():
    print(f"  {row['Modelo']:15s}: Diferença = {row['Diferenca']:.4f}")

print("\n" + "="*80)
print("ANÁLISE DE REDES NEURAIS CONCLUÍDA")
print("="*80)
