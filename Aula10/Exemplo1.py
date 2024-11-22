import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


try:
    print('Obtendo dados...')

    ENDERECO_DADOS = 'https://www.ispdados.rj.gov.br/Arquivos/BaseDPEvolucaoMensalCisp.csv'

    df_ocorrencias = pd.read_csv(ENDERECO_DADOS, sep=';', encoding='iso-8859-1')
    df_roubo_veiculo = df_ocorrencias[['munic', 'roubo_veiculo']]
    df_roubo_veiculo = df_roubo_veiculo.groupby(['munic']).sum(['roubo_veiculo']).reset_index()

    print(df_roubo_veiculo.head())
    print('\nDados obtidos com sucesso!')

except ImportError as e:
    print(f'Erro ao obter dados: {e}')
    exit()    

try:
    # print('\nCalculando informações sobre padrão de roubo de veículos...')
    array_roubo_veiculo = np.array(df_roubo_veiculo['roubo_veiculo'])
    media_roubo_veiculo = np.mean(array_roubo_veiculo)
    mediana_roubo_veiculo = np.median(array_roubo_veiculo)
    distancia_media_mediana = abs((media_roubo_veiculo - mediana_roubo_veiculo) / mediana_roubo_veiculo)

    maximo = np.max(array_roubo_veiculo)
    minimo = np.min(array_roubo_veiculo)
    amplitude_total = maximo - minimo

    q1 = np.quantile(array_roubo_veiculo, 0.25, method='weibull')
    q2 = np.quantile(array_roubo_veiculo, 0.50, method='weibull')
    q3 = np.quantile(array_roubo_veiculo, 0.75, method='weibull')

    iqr = q3 - q1

    limite_superior = q3 + (1.5 * iqr)
    limite_inferior = q1 - (1.5 * iqr)

    #FILTRANDO OS OUTLIERS
    df_roubo_veiculo_outliers_inferiores = df_roubo_veiculo[df_roubo_veiculo['roubo_veiculo'] < limite_inferior]
    df_roubo_veiculo_outliers_superiores = df_roubo_veiculo[df_roubo_veiculo['roubo_veiculo'] > limite_superior]

    print('\nMEDIDAS DE TENDÊNCIA CENTRAL:')
    print(30*'-')
    print(f'Média: {media_roubo_veiculo:.2f}')
    print(f'Mediana: {mediana_roubo_veiculo:.2f}')
    print(f'Distância: {distancia_media_mediana:.2f}')

    print('\nMEDIDAS DE DISPERSÃO:')
    print(30*'-')
    print('Máximo: ', maximo)
    print('Mínimo: ', minimo)
    print('Amplitude Total: ', amplitude_total)

    print('\nMEDIDAS DE POSIÇÃO:')
    print(30*'-')
    print('Mínimo: ', minimo)
    print(f'Limite Inferior: {limite_inferior:.2f}')
    print('Q1 (25%): ', q1)
    print('Q2 (50%): ', q2)
    print('Q3 (75%): ', q3)
    print(f'IQR: {iqr:.2f}')
    print(f'Limite Superior: {limite_superior:.2f}')
    print('Máximo: ', maximo)

    print('\nMunicípios com outliers inferiores:')
    print(30*'-')
    if len(df_roubo_veiculo_outliers_inferiores) == 0:
        print('Não existem outliers inferiores!')
    else:
        print(df_roubo_veiculo_outliers_inferiores.sort_values(by='roubo_veiculo', ascending=True))
    print('\nMunicípios com outliers superiores:')
    print(30*'-')
    if len(df_roubo_veiculo_outliers_superiores) == 0:
        print('Não existem outliers superiores!')
    else:
        print(df_roubo_veiculo_outliers_superiores.sort_values(by='roubo_veiculo', ascending=False))
    

except ImportError as e:
    print(f'Erro ao obter informações sobre padrão de roubo de veículos: {e}')
    exit()

try:
    plt.subplots(1, 2, figsize=(16, 7))
    plt.suptitle('Análise de roubo de veículos no RJ')
    plt.subplot(1, 2, 1)    
    plt.boxplot(array_roubo_veiculo, vert=False, showmeans=True)
    plt.title('Boxplot dos Dados')
    plt.subplot(1, 2, 2)
    plt.text(0.1, 0.9, f'Média: {media_roubo_veiculo:.2f}', fontsize=12)
    plt.text(0.1, 0.8, f'Mediana: {mediana_roubo_veiculo:.2f}', fontsize=12)
    plt.text(0.1, 0.7, f'Distância: {distancia_media_mediana:.2f}', fontsize=12)
    plt.text(0.1, 0.6, f'Menor valor: {minimo}', fontsize=12) 
    plt.text(0.1, 0.5, f'Limite inferior: {limite_inferior}', fontsize=12)
    plt.text(0.1, 0.4, f'Q1: {q1}', fontsize=12)
    plt.text(0.1, 0.3, f'Q3: {q3}', fontsize=12)
    plt.text(0.1, 0.2, f'Limite superior: {limite_superior}', fontsize=12)
    plt.text(0.1, 0.1, f'Maior valor: {maximo}', fontsize=12)
    plt.text(0.1, 0.0, f'Amplitude Total: {amplitude_total}', fontsize=12)
    plt.title('Medidas Observadas')
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()

except ImportError as e:
    print(f'Erro ao visualizar dados: {e}')
    exit()