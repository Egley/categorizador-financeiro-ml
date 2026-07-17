# =============================================================
# CAPSTONE — Categorizador de transações financeiras
# Modelo supervisionado de classificação de texto multiclasse
# Égley + Claude · Dia 9 do mini curso de ML
# =============================================================

import pandas as pd
import unicodedata

# -------------------------------------------------------------
# ETAPA 1 — CARREGAR A BASE  (o famoso pd.read_excel)
# -------------------------------------------------------------
df = pd.read_excel('base_treino_ML.xlsx', sheet_name='Base_v3')
df.columns = [c.strip() for c in df.columns]   # remove espaços acidentais nos nomes

# -------------------------------------------------------------
# ETAPA 2 — PRÉ-PROCESSAMENTO  (limpeza que discutimos no Dia 4)
# minúsculas + sem acentos: "Saúde" e "saude" viram a mesma coisa
# -------------------------------------------------------------
def limpar(s):
    s = str(s).lower().strip()
    return unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode()

df['_texto'] = df['Identificação (Merchant)'].apply(limpar)  # a feature estrela
df['_valor'] = pd.to_numeric(df['Valor'], errors='coerce').fillna(0)
df['_pgto']  = df['Pgto'].astype(str).str.strip().str.lower()
df['_tipo']  = df['Tipo'].astype(str).str.strip().str.lower()

# -------------------------------------------------------------
# ETAPA 3 — DEFINIR X (features) E y (target)   [Dia 4]
# -------------------------------------------------------------
X = df[['_texto', '_valor', '_pgto', '_tipo']]
y = df['Categoria'].astype(str).str.strip()

# -------------------------------------------------------------
# ETAPA 4 — DIVISÃO TREINO/TESTE   [Dia 5]
# 80/20, aleatória. stratify=y garante que cada categoria mantém
# sua proporção nos dois lados (protege as classes magras).
# random_state=42: "trava" o sorteio p/ o resultado ser reproduzível.
# -------------------------------------------------------------
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------------------------------------
# ETAPA 5 — TRANSFORMAR AS FEATURES EM NÚMEROS   [Dia 4: "modelos
# só entendem números"]
#
# - Texto -> TF-IDF: cada pedaço de texto vira uma coluna numérica
#   com o "peso" daquele pedaço. Usamos n-gramas de CARACTERES
#   (2 a 4 letras): "farm", "uber", "saco"... Isso deixa o modelo
#   reconhecer padrões mesmo em merchants nunca vistos que
#   contenham pedaços conhecidos (generalização!).
# - Categóricas (pgto, tipo) -> One-Hot: cada opção vira coluna 0/1.
# - Valor -> StandardScaler: coloca na mesma escala das outras.
# -------------------------------------------------------------
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

preparador = ColumnTransformer([
    ('txt', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4)), '_texto'),
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['_pgto', '_tipo']),
    ('num', StandardScaler(), ['_valor']),
])

# Versão p/ Naive Bayes (ele exige valores não-negativos -> sem o scaler)
preparador_nb = ColumnTransformer([
    ('txt', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4)), '_texto'),
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['_pgto', '_tipo']),
])

# -------------------------------------------------------------
# ETAPA 6 — OS TRÊS CANDIDATOS   [Dia 8]
# Pipeline = preparação + modelo empacotados juntos (evita vazamento:
# o preparador "aprende" só com o treino).
# class_weight='balanced' = o "peso extra às classes raras" que
# conversamos (hiperparâmetro contra o desbalanceamento).
# n_estimators=300 = 300 árvores na floresta.
# -------------------------------------------------------------
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

candidatos = {
    'Regressão Logística': Pipeline([
        ('prep', preparador),
        ('modelo', LogisticRegression(max_iter=2000, class_weight='balanced')),
    ]),
    'Naive Bayes': Pipeline([
        ('prep', preparador_nb),
        ('modelo', MultinomialNB()),
    ]),
    'Random Forest': Pipeline([
        ('prep', preparador),
        ('modelo', RandomForestClassifier(
            n_estimators=300, class_weight='balanced', random_state=42)),
    ]),
}

# -------------------------------------------------------------
# ETAPA 7 — TREINAR E AVALIAR   [Dias 5, 6 e 7]
# .fit(X, y)  = treinar  |  .predict(X) = prever
# Comparamos treino vs teste (diagnóstico de overfitting) e
# F1-macro (média das classes com peso igual — desmascara o
# abandono das classes magras).
# -------------------------------------------------------------
from sklearn.metrics import accuracy_score, f1_score, classification_report

for nome, pipe in candidatos.items():
    pipe.fit(X_train, y_train)                      # <-- O TREINO
    pred_teste = pipe.predict(X_test)               # <-- A PROVA
    print(f"\n===== {nome} =====")
    print("Acurácia treino:", round(accuracy_score(y_train, pipe.predict(X_train)), 3))
    print("Acurácia teste :", round(accuracy_score(y_test, pred_teste), 3))
    print("F1-macro teste :", round(f1_score(y_test, pred_teste, average='macro'), 3))

# -------------------------------------------------------------
# ETAPA 8 — RAIO-X DO CAMPEÃO   [Dia 7]
# precision/recall por classe + matriz de confusão
# -------------------------------------------------------------
campea = candidatos['Random Forest']
pred = campea.predict(X_test)
print(classification_report(y_test, pred, zero_division=0))

# -------------------------------------------------------------
# ETAPA 9 — O PORTÃO DE REVISÃO   [seu design: human-in-the-loop]
# predict_proba devolve a probabilidade de cada classe.
# Confiança = probabilidade da classe vencedora.
# Abaixo de 80% -> linha marcada "revisar".
# -------------------------------------------------------------
probas = campea.predict_proba(X_test)
confianca = probas.max(axis=1)
revisar = confianca < 0.80

aceitas_certas = (pred[~revisar] == y_test.values[~revisar]).mean()
print(f"\nMarcadas p/ revisar: {revisar.sum()} de {len(pred)} ({revisar.mean():.0%})")
print(f"Acurácia nas aceitas automaticamente: {aceitas_certas:.1%}")
