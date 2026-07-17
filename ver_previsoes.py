# Roda o capstone inteiro (treina tudo de novo) e aproveita os objetos
exec(open('capstone_categorizador.py', encoding='utf-8').read())

# O modelo campeão prevê a BASE INTEIRA
pred_tudo = campea.predict(X)
proba_tudo = campea.predict_proba(X)
confianca_tudo = proba_tudo.max(axis=1)

# Monta a visão lado a lado
resultado = df[['Descrição', 'Identificação (Merchant)', 'Valor', 'Categoria']].copy()
resultado['Categoria_modelo'] = pred_tudo
resultado['Confianca'] = (confianca_tudo * 100).round(1)
resultado['Status'] = 'aceita_automatico'
resultado.loc[confianca_tudo < 0.80, 'Status'] = 'REVISAR'
resultado.loc[resultado['Categoria'] != resultado['Categoria_modelo'], 'Status'] = 'DISCORDA'

# Salva em Excel para você fuçar com filtros
resultado.to_excel('base_categorizada_pelo_modelo.xlsx', index=False)
print("Arquivo salvo: base_categorizada_pelo_modelo.xlsx")
print(resultado['Status'].value_counts())