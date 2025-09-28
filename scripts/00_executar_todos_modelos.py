# -*- coding: utf-8 -*-
"""
UNIVERSIDADE DE SÃO PAULO
MBA DATA SCIENCE & ANALYTICS USP/ESALQ
PREDIÇÃO DE RECOLHIMENTO DE VEÍCULOS - EXECUÇÃO DE TODOS OS MODELOS
TCC - Pipeline ML Motto

@author: [Seu Nome]
@date: 2025
"""

#%% Importação dos pacotes

import os
import sys
import subprocess
import time
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

#%% Configurações

# Diretórios dos modelos
modelos_dir = Path(".")
scripts = [
    "logit/01_logit_recolhimento.py",
    "arvore/02_arvore_recolhimento.py", 
    "xgboost/03_xgboost_recolhimento.py",
    "ensemble/04_ensemble_recolhimento.py",
    "redes_neurais/05_redes_neurais_recolhimento.py",
    "comparacao/06_comparacao_final.py",
    "07_insights_finais.py"
]

#%% Função para executar script

def executar_script(script_path):
    """Executar um script Python"""
    print(f"\n{'='*80}")
    print(f"EXECUTANDO: {script_path}")
    print(f"{'='*80}")
    
    try:
        # Executar script de dentro da subpasta
        script_path_obj = Path(script_path)
        script_dir = modelos_dir / script_path_obj.parent
        script_name = script_path_obj.name
        
        # Tratamento especial para script de insights (executa na raiz)
        if script_name == "07_insights_finais.py":
            script_dir = modelos_dir
            print("📊 Executando análise de insights finais...")
        
        result = subprocess.run([
            sys.executable, script_name
        ], capture_output=True, text=True, cwd=script_dir)
        
        if result.returncode == 0:
            print(f"✅ {script_path} executado com sucesso!")
            if result.stdout:
                print("Saída:")
                print(result.stdout[-500:])  # Últimas 500 caracteres
        else:
            print(f"❌ Erro ao executar {script_path}")
            print("Erro:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Exceção ao executar {script_path}: {e}")
        return False
    
    return True

#%% Função principal

def main():
    """Executar todos os modelos em sequência"""
    
    print("="*80)
    print("EXECUÇÃO DE TODOS OS MODELOS - TCC PREDIÇÃO DE RECOLHIMENTO")
    print("="*80)
    print(f"Data: {time.strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"Diretório: {modelos_dir.absolute()}")
    
    # Verificar se os scripts existem
    scripts_existentes = []
    for script in scripts:
        script_path = modelos_dir / script
        if script_path.exists():
            scripts_existentes.append(script)
        else:
            print(f"⚠️  Script não encontrado: {script}")
    
    print(f"\nScripts encontrados: {len(scripts_existentes)}")
    
    # Executar scripts em sequência
    resultados = {}
    tempo_inicio = time.time()
    
    for i, script in enumerate(scripts_existentes, 1):
        print(f"\n{'='*60}")
        print(f"EXECUTANDO {i}/{len(scripts_existentes)}: {script}")
        print(f"{'='*60}")
        
        tempo_script_inicio = time.time()
        sucesso = executar_script(script)
        tempo_script_fim = time.time()
        
        resultados[script] = {
            'sucesso': sucesso,
            'tempo': tempo_script_fim - tempo_script_inicio
        }
        
        if sucesso:
            print(f"⏱️  Tempo de execução: {tempo_script_fim - tempo_script_inicio:.1f} segundos")
        else:
            print(f"❌ Falha na execução após {tempo_script_fim - tempo_script_inicio:.1f} segundos")
    
    # Resumo final
    tempo_total = time.time() - tempo_inicio
    
    print(f"\n{'='*80}")
    print("RESUMO DA EXECUÇÃO")
    print(f"{'='*80}")
    
    sucessos = sum(1 for r in resultados.values() if r['sucesso'])
    falhas = len(resultados) - sucessos
    
    print(f"Total de scripts: {len(resultados)}")
    print(f"Sucessos: {sucessos}")
    print(f"Falhas: {falhas}")
    print(f"Tempo total: {tempo_total:.1f} segundos ({tempo_total/60:.1f} minutos)")
    
    print(f"\nDetalhes por script:")
    for script, resultado in resultados.items():
        status = "✅" if resultado['sucesso'] else "❌"
        tempo = resultado['tempo']
        print(f"  {status} {script:30s} - {tempo:6.1f}s")
    
    # Salvar log
    log_df = pd.DataFrame([
        {
            'script': script,
            'sucesso': resultado['sucesso'],
            'tempo_segundos': resultado['tempo']
        }
        for script, resultado in resultados.items()
    ])
    
    log_df.to_csv('log_execucao_modelos.csv', index=False)
    print(f"\nLog salvo em: log_execucao_modelos.csv")
    
    if falhas == 0:
        print(f"\n🎉 TODOS OS MODELOS EXECUTADOS COM SUCESSO!")
        print(f"Verifique os resultados em cada pasta de modelo.")
        
        # Mostrar insights finais se disponíveis
        insights_dir = Path("outputs/insights_finais")
        if insights_dir.exists():
            print(f"\n📊 INSIGHTS FINAIS GERADOS:")
            print(f"{'='*50}")
            for arquivo in insights_dir.glob("*"):
                print(f"  📄 {arquivo.name}")
            
            # Mostrar resumo do relatório principal
            relatorio_path = insights_dir / "relatorio_insights_finais.md"
            if relatorio_path.exists():
                print(f"\n📋 Resumo do Relatório Principal:")
                with open(relatorio_path, 'r', encoding='utf-8') as f:
                    linhas = f.readlines()
                    for i, linha in enumerate(linhas[:20]):  # Primeiras 20 linhas
                        if linha.strip():
                            print(f"  {linha.strip()}")
                    if len(linhas) > 20:
                        print(f"  ... (arquivo completo em: {relatorio_path})")
    else:
        print(f"\n⚠️  {falhas} script(s) falharam. Verifique os erros acima.")
    
    print(f"\n{'='*80}")
    print("EXECUÇÃO CONCLUÍDA")
    print(f"{'='*80}")

#%% Executar se chamado diretamente

if __name__ == "__main__":
    main()



