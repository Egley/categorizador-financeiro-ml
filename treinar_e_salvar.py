# =============================================================
# KIT DE PRODUÇÃO — parte 1: TREINAR E SALVAR
# Rode este script: (a) agora, 1 vez; (b) a cada retreino
# (sugestão: trimestral, ou quando acumular muitas correções).
#
# O que ele faz: treina o Random Forest para os 3 targets
# (Categoria, CC, Origem) usando a base rotulada, avalia cada um,
# e salva os modelos prontos em arquivos .joblib.
#
# Requisitos: base_treino_ML.xlsx na mesma pasta (aba Base_v3).
# =============================================================
import pandas as pd
import unicodedata
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

ARQUIVO_BASE = 'base_treino_ML.xlsx'
ABA_BASE = 'Base_v3'
TARGETS = ['Categoria', 'CC', 'Origem (Responsável)']

def limpar(s):
    s = str(s).lower().strip()
    return unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode()

def preparar_features(df):
    """Transforma o dataframe cru nas 4 features do modelo."""
    out = pd.DataFrame()
    out['_texto'] = df['Identificação (Merchant)'].apply(limpar)
    out['_valor'] = pd.to_numeric(df['Valor'], errors='coerce').fillna(0)
    out['_pgto']  = df['Pgto'].astype(str).str.lower().str.strip()
    out['_tipo']  = df['Tipo'].astype(str).str.lower().str.strip()
    return out

def novo_preparador():
    return ColumnTransformer([
        ('txt', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4)), '_texto'),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['_pgto', '_tipo']),
        ('num', StandardScaler(), ['_valor']),
    ])

print('Carregando a base...')
df = pd.read_excel(ARQUIVO_BASE, sheet_name=ABA_BASE)
df.columns = [c.strip() for c in df.columns]
X = preparar_features(df)
print(f'{len(df)} linhas carregadas.')

for target in TARGETS:
    y = df[target].astype(str).str.strip()

    # avaliação honesta (treino/teste) só para você acompanhar a saúde do modelo
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    aval = Pipeline([('prep', novo_preparador()),
                     ('modelo', RandomForestClassifier(
                         n_estimators=300, class_weight='balanced', random_state=42))])
    aval.fit(X_tr, y_tr)
    pred = aval.predict(X_te)
    print(f'\n{target}: acuracia teste={accuracy_score(y_te, pred):.3f} '
          f'| F1-macro={f1_score(y_te, pred, average="macro"):.3f}')

    # modelo FINAL de produção: treinado com 100% dos dados
    # (depois de avaliado, não faz sentido desperdiçar os 20% do teste)
    final = Pipeline([('prep', novo_preparador()),
                      ('modelo', RandomForestClassifier(
                          n_estimators=300, class_weight='balanced', random_state=42))])
    final.fit(X, y)
    nome_arquivo = f'modelo_{limpar(target).replace(" ", "_").replace("(", "").replace(")", "")}.joblib'
    joblib.dump(final, nome_arquivo)
    print(f'  -> salvo em {nome_arquivo}')

print('\nTreino concluido! Os arquivos .joblib sao os "cerebros" prontos.')
