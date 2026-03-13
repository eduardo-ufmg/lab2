import numpy as np
import pandas as pd


def us_para_ms(us):
    return us * 1e-3  # 1 us = 1e-3 ms


def computa_media_desvio(data):
    media = np.mean(data)
    desvio_padrao = np.std(data, ddof=1)  # Usando ddof=1 para amostra
    return media, desvio_padrao


def le_dados_csv(caminho_arquivo):
    df = pd.read_csv(caminho_arquivo)
    valores_ms = us_para_ms(
        df.iloc[:, 0].to_numpy()
    )  # Os dados estão na primeira coluna
    return valores_ms


def constroi_csv_periodos_medidos(caminho_saida, caminhos_entrada):
    periodos_medidos = []

    for configuracao, caminho in caminhos_entrada.items():
        dados_ms = le_dados_csv(caminho)
        media, desvio = computa_media_desvio(dados_ms)
        periodos_medidos.append((configuracao, media, desvio))

    df_saida = pd.DataFrame(
        periodos_medidos, columns=["Configuracao (ms)", "Media (ms)", "Desvio Padrao (ms)"]
    )
    df_saida.to_csv(caminho_saida, index=False)


if __name__ == "__main__":
    caminhos_entrada = {
        1: "Tempo_1ms.txt",
        10: "Tempo_10ms.txt",
        100: "Tempo_100ms.txt",
    }

    caminho_saida = "Periodos_Medidos.csv"
    constroi_csv_periodos_medidos(caminho_saida, caminhos_entrada)
