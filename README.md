# Predição de Recolhimento de Veículos em Serviços de Mobilidade Urbana

## 📋 Descrição do Projeto

Este repositório contém a implementação completa de um sistema de predição para recolhimento de veículos utilizando técnicas de Machine Learning. O projeto foi desenvolvido como Trabalho de Conclusão de Curso (TCC) e implementa diversos algoritmos para classificação binária com validação temporal rigorosa.

## 🎯 Objetivos

- **Objetivo Principal**: Desenvolver um modelo preditivo para identificar a necessidade de recolhimento de veículos
- **Objetivos Específicos**:
  - Implementar pipeline de feature engineering temporal
  - Desenvolver e comparar múltiplos algoritmos de ML
  - Implementar validação temporal robusta
  - Prevenir data leakage em dados temporais
  - Analisar interpretabilidade dos modelos

## 🛠️ Tecnologias Utilizadas

- **Python 3.8+**
- **Principais Bibliotecas**:
  - scikit-learn (modelos tradicionais)
  - XGBoost (gradient boosting)
  - TensorFlow/Keras (redes neurais)
  - pandas, numpy (manipulação de dados)
  - matplotlib, seaborn (visualização)
  - SHAP (interpretabilidade)

## 🚀 Como Executar

### 1. Instalação das Dependências

```bash
pip install -r requirements.txt
```

### 2. Estrutura de Execução

```bash
# Executar todos os modelos em sequência
python scripts/00_executar_todos_modelos.py

# Ou executar modelos individuais:
python scripts/modelos/01_logit/01_logit_recolhimento.py
python scripts/modelos/02_arvore/02_arvore_recolhimento.py
python scripts/modelos/03_xgboost/03_xgboost_recolhimento.py
python scripts/modelos/04_ensemble/04_ensemble_recolhimento.py
python scripts/modelos/05_redes_neurais/05_redes_neurais_recolhimento.py

# Comparação final e análises
python scripts/modelos/07_comparacao/06_comparacao_final.py
python scripts/06_insights_finais.py
```

### 3. Estrutura dos Dados

Os scripts utilizam o dataset em formato Parquet:
- **Dataset Principal**: `data/objective_dataset_clima_*.parquet`
- **Target**: `recolhimento_evento` (binário: 0/1)
- **Features**: 60 variáveis incluindo temporais, climáticas e operacionais
- **Splits**: Dados pré-divididos em `data/splits/`

## 📊 Algoritmos Implementados

### 01. Regressão Logística
- Modelo simples e stepwise
- Análise de coeficientes e significância

### 02. Árvores de Decisão
- Árvore simples, podada e otimizada
- Visualização da estrutura de decisão

### 03. XGBoost
- Versões: simples, otimizado, temporal, early stopping
- Análise de feature importance

### 04. Métodos Ensemble
- Voting (soft/hard), Bagging, AdaBoost
- Gradient Boosting, Stacking
- Análise de correlação entre modelos

### 05. Redes Neurais
- MLP (simples e profundo)
- LSTM para dados sequenciais
- Análise de overfitting

### 06. Análises Consolidadas
- Comparação de performance
- Análise de complexidade vs performance
- Trade-offs operacionais

## 📈 Principais Resultados

- **Melhor Modelo**: XGBoost Otimizado (AUC: 0.9174)
- **Validação Temporal**: TimeSeriesSplit com 5 folds
- **Features Críticas**: `total_7d`, `dias_desde_ultima_manutencao_eletrica`
- **Descoberta Principal**: Correlação fraca entre complexidade algorítmica e performance (r=0.201)

## 🔍 Análises Especializadas

- **Prevenção de Data Leakage**: Split temporal rigoroso
- **Interpretabilidade**: SHAP values e feature importance
- **Robustez Temporal**: Validação em diferentes períodos
- **Trade-offs Operacionais**: Análise de custos FP vs FN
- **Análise de Sensibilidade**: Múltiplas métricas de robustez

## 📝 Estrutura de Dados

### Dataset Principal
- **Registros**: 15.132 serviços de manutenção
- **Período**: 9 meses (jan-set 2024)
- **Features**: 60 variáveis numéricas
- **Target**: 25.7% de casos positivos (recolhimento)

### Tipos de Features
- **Temporais de Recência** (16): Dias desde última manutenção por subsistema
- **Janelas Temporais** (21): Contagens em 7, 30 e 90 dias
- **Operacionais** (8): Quilometragem, flags, sintomas
- **Climáticas** (8): Temperatura, precipitação, umidade
- **Categóricas** (7): Classificação, sazonalidade

## 👥 Autor

**Lucas Araújo**  
Trabalho de Conclusão de Curso (TCC) - MBA em Data Science