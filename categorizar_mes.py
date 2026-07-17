# =============================================================
# KIT DE PRODUÇÃO — parte 2: CATEGORIZAR O MÊS (o script do dia a dia)
#
# O que ele faz:
#   1. Lê as transações novas (extrato_novo.xlsx)
#   2. Aplica o MAPEAMENTO (aba da base) p/ normalizar descrição -> merchant
#   3. Carrega os 3 modelos salvos e prevê Categoria, CC e Origem
#   4. Marca cada linha: aceita_automatico ou REVISAR (confiança < 80%
#      em qualquer um dos 3, merchant sem regra, ou marketplace)
#   5. Salva tudo em extrato_categorizado.xlsx
#
# Requisitos na pasta: os 3 .joblib (gerados pelo treinar_e_salvar.py),
# base_treino_ML.xlsx (p/ ler a aba Mapeamento) e extrato_novo.xlsx.
#
# O extrato_novo.xlsx precisa ter as colunas: Descrição, Valor, Pgto, Tipo
# =============================================================
import pandas as pd
import unicodedata
import joblib

ARQUIVO_NOVO = 'extrato_novo.xlsx'
ARQUIVO_MAPEAMENTO = 'base_treino_ML.xlsx'   # aba Mapeamento
ARQUIVO_SAIDA = 'extrato_categorizado.xlsx'
THRESHOLD = 0.80                              # portão de revisão
MARKETPLACES = {'amazon', 'mercado livre', 'americanas', 'magalu',
                'tiktok shop', 'shopee'}      # sempre revisar (ambiguidade real)

def limpar(s):
    s = str(s).lower().strip()
    return unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode()

# ---- 1. transações novas -------------------------------------------------
novas = pd.read_excel(ARQUIVO_NOVO)
novas.columns = [c.strip() for c in novas.columns]
print(f'{len(novas)} transacoes novas carregadas.')

# ---- 2. normalização via mapeamento -------------------------------------
mapa = pd.read_excel(ARQUIVO_MAPEAMENTO, sheet_name='Mapeamento')
mapa.columns = [c.strip() for c in mapa.columns]
col_padrao, col_ident = mapa.columns[0], mapa.columns[1]
regras = [(limpar(p), str(i).strip())
          for p, i in zip(mapa[col_padrao], mapa[col_ident])
          if pd.notna(p) and pd.notna(i)]

def normalizar(descricao):
    d = limpar(descricao)
    for padrao, ident in regras:
        if padrao in d:
            return ident, True
    return descricao.strip(), False           # sem regra: mantém a descrição

resultado = novas.copy()
norm = novas['Descrição'].apply(normalizar)
resultado['Merchant'] = [m for m, _ in norm]
resultado['Tem_regra'] = [t for _, t in norm]
print(f"Mapeamento cobriu {resultado['Tem_regra'].mean():.0%} das linhas.")

# ---- 3. features + previsões dos 3 modelos -------------------------------
X = pd.DataFrame()
X['_texto'] = resultado['Merchant'].apply(limpar)
X['_valor'] = pd.to_numeric(resultado['Valor'], errors='coerce').fillna(0)
X['_pgto']  = resultado['Pgto'].astype(str).str.lower().str.strip()
X['_tipo']  = resultado['Tipo'].astype(str).str.lower().str.strip()

modelos = {
    'Categoria': 'modelo_categoria.joblib',
    'CC': 'modelo_cc.joblib',
    'Origem (Responsável)': 'modelo_origem_responsavel.joblib',
}
menor_confianca = pd.Series(1.0, index=resultado.index)
for alvo, arquivo in modelos.items():
    modelo = joblib.load(arquivo)
    resultado[f'{alvo}_sugerida'] = modelo.predict(X)
    conf = modelo.predict_proba(X).max(axis=1)
    resultado[f'{alvo}_confianca'] = (conf * 100).round(1)
    menor_confianca = pd.concat([menor_confianca, pd.Series(conf)], axis=1).min(axis=1)

# ---- 4. o portão de revisão ----------------------------------------------
eh_marketplace = resultado['Merchant'].apply(lambda m: limpar(m) in MARKETPLACES)
revisar = (menor_confianca < THRESHOLD) | (~resultado['Tem_regra']) | eh_marketplace
resultado['Status'] = 'aceita_automatico'
resultado.loc[revisar, 'Status'] = 'REVISAR'

# ---- 5. salvar ------------------------------------------------------------
resultado.to_excel(ARQUIVO_SAIDA, index=False)
print(f"\nSalvo: {ARQUIVO_SAIDA}")
print(resultado['Status'].value_counts().to_string())
print('\nFluxo: filtre por REVISAR no Excel, corrija o necessario, e '
      'adicione as linhas confirmadas na base de treino p/ o proximo retreino.')
