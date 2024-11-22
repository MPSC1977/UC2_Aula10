import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


try:
    print('Obtendo dados...')

    ENDERECO_DADOS = 'https://www.ispdados.rj.gov.br/Arquivos/BaseDPEvolucaoMensalCisp.csv'

    df_ocorrencias = pd.read_csv(
        ENDERECO_DADOS, sep=';', encoding='iso-8859-1')

    df_veiculos_recuperados = df_ocorrencias[['cisp', 'recuperacao_veiculos']]

    df_veiculos_recuperados = df_veiculos_recuperados.groupby(['cisp']).sum(['recuperacao_veiculos']).reset_index()

    print(df_veiculos_recuperados.head())
    print('Dados obtidos com sucesso!')

except ImportError as e:
    print(f'Erro ao obter dados: {e}')
    exit()

try:
    array_veiculos_recuperados = np.array(df_veiculos_recuperados['recuperacao_veiculos'])
    
    media_veiculos_recuperados = np.mean(array_veiculos_recuperados)
    mediana_veiculos_recuperados = np.median(array_veiculos_recuperados)
    distancia = abs((media_veiculos_recuperados - mediana_veiculos_recuperados) / mediana_veiculos_recuperados) * 100

    maximo = np.max(array_veiculos_recuperados)
    minimo = np.min(array_veiculos_recuperados)
    amplitude = maximo - minimo

    q1 = np.quantile(array_veiculos_recuperados, 0.25, method='weibull')
    q2 = np.quantile(array_veiculos_recuperados, 0.50, method='weibull')
    q3 = np.quantile(array_veiculos_recuperados, 0.75, method='weibull')
    iqr = q3 - q1
    limite_superior = q3 + (1.5 * iqr)
    limite_inferior = q1 - (1.5 * iqr)

    df_veiculos_rec_outliers_inferiores = df_veiculos_recuperados[df_veiculos_recuperados['recuperacao_veiculos'] < limite_inferior]
    df_veiculos_rec_outliers_superiores = df_veiculos_recuperados[df_veiculos_recuperados['recuperacao_veiculos'] > limite_superior]

    print('\nMEDIDAS DE TENDÊNCIA CENTRAL')
    print(30*'=')
    print(f'A média de veículos recuperados é: {media_veiculos_recuperados:.2f}')
    print(f'A mediana de veículos recuperados é: {mediana_veiculos_recuperados:.2f}')
    print(f'A distância entre a média e a mediana de veículos recuperados é: {distancia:.2f}%')
    
    print('\nMEDIDAS DE DISPERSÃO')
    print(20*'=')
    print(f'Máximo: {maximo}')
    print(f'Mínimo: {minimo}')
    print(f'Amplitude total: {amplitude}')

    print('\nMEDIDAS DE POSIÇÃO')
    print(20*'=')
    print(f'Mínimo: {minimo}')
    print(f'Limite inferior: {limite_inferior}')
    print(f'Q1: {q1}')
    print(f'Q2: {q2}')
    print(f'Q3: {q3}')
    print(f'IQR: {iqr}')
    print(f'Limite superior: {limite_superior}')
    print(f'Máximo: {maximo}')

    print('\nOUTLIERS POR DELEGACIA')
    print(25*'=')

    print('\nOutliers inferiores')
    print(20*'=')
    if len(df_veiculos_rec_outliers_inferiores) == 0:
        print('Não existem outliers inferiores!')
    else:
        print(df_veiculos_rec_outliers_inferiores.sort_values(by='recuperacao_veiculos', ascending=True))

    print('\nOutliers superiores: ')
    print(20*'=')
    if len(df_veiculos_rec_outliers_superiores) == 0:
        print('Não existe outliers superiores!')
    else:
        print(df_veiculos_rec_outliers_superiores.sort_values(by='recuperacao_veiculos', ascending=False))

    print('\nCONCLUSÃO DA ANÁLISE: ')
    print('\nVerificou-se uma grande dispersão de dados, dessa forma pode-se afirmar que não há um padrão no número de recuperação de veículos nas diferentes CISPS. Sendo assim, observa-se delegacias que tiverem ocorrências de recuperação muito mais expressivas do que outras, conforme abaixo: ')
    print('\nTOP 5 DELEGACIAS COM MAIOR ÍNDICE DE RECUPERAÇÃO DE VEÍCULOS: ')
    print(60*'=')
    print(df_veiculos_rec_outliers_superiores.sort_values(by='recuperacao_veiculos', ascending=False).head(5))
    print('\nTOP 5 DELEGACIAS COM MENOR ÍNDICE DE RECUPERAÇÃO DE VEÍCULOS: ')
    print(60*'=')
    print(df_veiculos_recuperados.sort_values(by='recuperacao_veiculos', ascending=False).tail(5))

except ImportError as e:
    print(f'Erro ao obter informações sobre recuperação de veículos: {e}')
    exit()

try:
    print('Calculando medidas de distribuição...')

    assimetria = df_veiculos_recuperados['recuperacao_veiculos'].skew()
    curtose = df_veiculos_recuperados['recuperacao_veiculos'].kurtosis()

    print('\nMedidas de distribuição: ')
    print(30*'-')
    print(f'Assimetria: {assimetria:.2f}')
    print(f'Curtose: {curtose:.2f}')

except ImportError as e:
    print(f'Erro ao calcular medidas de distribuição: {e}')
    exit()

try:
    plt.subplots(2, 2, figsize=(16, 7))
    plt.suptitle('Análise de recuperação de veículos no RJ')

    plt.subplot(2, 2, 1)    
    plt.boxplot(array_veiculos_recuperados, vert=False, showmeans=True)
    plt.title('Recuperação de Veículos')

    plt.subplot(2, 2, 2)
    plt.hist(array_veiculos_recuperados, bins=50, edgecolor='black')
    plt.axvline(media_veiculos_recuperados, color='g', linewidth=1)
    plt.axvline(mediana_veiculos_recuperados, color='y', linewidth=1)
    plt.title('Histograma Recuperação de Veículos')

    plt.subplot(2, 2, 3)
    plt.text(0.1, 0.9, f'Média: {media_veiculos_recuperados:.2f}', fontsize=12)
    plt.text(0.1, 0.8, f'Mediana: {mediana_veiculos_recuperados:.2f}', fontsize=12)
    plt.text(0.1, 0.7, f'Distância: {distancia:.2f}', fontsize=12)
    plt.text(0.1, 0.6, f'Menor valor: {minimo}', fontsize=12) 
    plt.text(0.1, 0.5, f'Limite inferior: {limite_inferior}', fontsize=12)
    plt.text(0.1, 0.4, f'Q1: {q1}', fontsize=12)
    plt.text(0.1, 0.3, f'Q3: {q3}', fontsize=12)
    plt.text(0.1, 0.2, f'Limite superior: {limite_superior}', fontsize=12)
    plt.text(0.1, 0.1, f'Maior valor: {maximo}', fontsize=12)
    plt.text(0.1, 0.0, f'Amplitude Total: {amplitude}', fontsize=12)
    plt.title('Medidas Observadas')

    plt.axis('off')
    plt.tight_layout()

    plt.subplot(2, 2, 4)
    plt.text(0.1, 0.5, f'Assimetria: {assimetria:.2f}', fontsize=12)
    plt.text(0.1, 0.4, f'Curtose: {curtose:.2f}', fontsize=12)
    plt.title('Impressão de Medidas Estatísticas')
    
    plt.axis('off')
    plt.tight_layout()
    
    plt.show()

except ImportError as e:
    print(f'Erro ao visualizar dados: {e}')
    exit()