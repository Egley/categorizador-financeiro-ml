# 💰 Categorizador Financeiro com Machine Learning

> **Sistema de categorização automática de transações bancárias pessoais** — classificação de texto multiclasse com Random Forest, fluxo de revisão por confiança (human-in-the-loop) e ciclo de retreino.

**Sobre a autoria deste projeto:** eu não sabia programar Python quando construí este sistema. Usei IA (Claude/Anthropic) como ferramenta de desenvolvimento — eu defini o problema, tomei todas as decisões de modelagem de dados e arquitetura, fiz o controle de qualidade da base, e a IA escreveu o código sob minha direção. Este projeto documenta, na prática, como a IA pode derrubar a barreira da curva de aprendizado técnico: **não dominar (ainda) uma linguagem deixou de ser impedimento para construir soluções reais.** Atualmente estou aprendendo Python reconstruindo cada script deste repositório linha por linha.

---

## O problema

Eu mantenho um controle financeiro pessoal detalhado, consolidando transações de múltiplos bancos com categorização em três dimensões: **Categoria** (supermercado, saúde, lazer...), **Centro de Custo** (essenciais / não essenciais / imóveis) e **Responsável** pelo gasto.

A categorização manual era o gargalo. A primeira automação — um dicionário de 130 regras "se contém X → categoria Y" num pipeline N8N — funcionava, mas não **generaliza**: todo estabelecimento novo exigia regra nova, escrita à mão, para sempre.

## A solução: um sistema híbrido em camadas

```
Extrato (OFX/Excel)
   │
   ▼
[1] NORMALIZAÇÃO POR REGRAS (209 regras)
    descrição bruta → merchant limpo ("MP*CINTIAESTETICA" → "cintia estetica")
   │
   ▼
[2] MODELOS DE ML (3 classificadores Random Forest)
    merchant + valor + forma de pgto + tipo → Categoria, CC e Responsável
    cada previsão sai com um grau de CONFIANÇA
   │
   ▼
[3] PORTÃO DE REVISÃO (human-in-the-loop)
    confiança ≥ 80% → aceita automático
    confiança < 80%, merchant sem regra ou marketplace → REVISAR
   │
   ▼
[4] REVISÃO HUMANA → correções alimentam a base → RETREINO periódico
```

Cada camada trata a incerteza que consegue: **regras para o que é fixo e conhecido; modelo para o que varia e cresce; humano para o que só o contexto resolve.**

## Decisões de modelagem (o coração do projeto)

Estas decisões nasceram de problemas reais encontrados nos dados:

- **Ambiguidade dissolvida por design.** Corridas de Uber podiam ser "essenciais" (médico, trabalho) ou "conforto" — informação que **não existe no extrato**. Em vez de pedir ao modelo para adivinhar o inadivinhável, redesenhei as categorias: "uber" virou categoria própria. Resultado: precision e recall de 100% nessa classe.
- **Entity resolution manual.** O mesmo estabelecimento aparecia com vários nomes: maquininhas diferentes do mesmo sacolão, prestadoras recebendo ora como pessoa física (Pix) ora como empresa (maquininha), cinco canais de recarga do bilhete único. Nenhum algoritmo conectaria "Yuri S L De Sacolao" a "Sacolao Popular" — só o conhecimento de domínio. Cada unificação virou regra permanente no mapeamento.
- **Prevenção de data leakage.** A primeira versão da coluna de identificação embutia a resposta na pista (ex.: "uber-conforto" já continha o centro de custo). Refatorei para a regra: **identificação = QUEM recebeu o dinheiro; categoria = O QUE significa** — o pulo semântico é exatamente o que o modelo deve aprender.
- **Marketplaces sempre em revisão.** Amazon, Mercado Livre e afins são fisicamente imprevisíveis pelo extrato (o mesmo "AMAZON*BR" pode ser lâmpada ou presente). O sistema não finge que sabe: marca para revisão, sempre.
- **Escopo dimensionado pelos dados.** Subcategorias existem na planilha para análise, mas ficaram fora do treino da v1: classes com 3-5 exemplos não se aprendem, se decoram. Entram na v2 quando o histórico engordar.

## Resultados (teste com 20% dos dados, nunca vistos no treino)

| Modelo | Acurácia | F1-macro |
|---|---|---|
| Categoria (12 classes) | 92,1% | 0,88 |
| Centro de Custo (3 classes) | 96,2% | 0,97 |
| Responsável (8 classes) | 92,9% | 0,86 |

**O resultado mais importante não é a acurácia — é o portão de revisão:** no conjunto de teste, as linhas aceitas automaticamente (80% do volume) tiveram **acurácia de 100%**, e **todos os erros do modelo caíram no grupo marcado para revisão**. Na prática: nenhum erro entra na base sem passar por um humano.

As classes mais fracas (lazer, vestuário) são explicadas pela cauda longa: 57% dos merchants aparecem uma única vez no histórico — bares e lojinhas de visita única, que o fluxo de revisão apara.

## Duelo de algoritmos

Seguindo o princípio "no free lunch", três candidatos foram treinados e comparados:

| Candidato | Acurácia teste | Observação |
|---|---|---|
| **Random Forest** ⭐ | **92,1%** | Campeão — aproveita as features tabulares além do texto |
| Regressão Logística | 90,0% | Forte em texto, probabilidades bem calibradas |
| Naive Bayes | 77,0% | Baseline; confiança mal calibrada furaria o portão de revisão |

## Stack e estrutura

- **Python** · pandas · scikit-learn (TF-IDF por n-gramas de caracteres + Random Forest com `class_weight='balanced'`) · joblib
- `treinar_e_salvar.py` — treina os 3 modelos e salva os artefatos (.joblib)
- `categorizar_mes.py` — o script do dia a dia: normaliza, prevê, aplica o portão e gera o Excel revisável
- Integra-se ao meu [pipeline de ingestão N8N](https://github.com/Egley/pipeline-finananceiro-n8n) (projeto anterior, em migração para Python puro)

*Os dados reais (transações pessoais) e os modelos treinados não estão no repositório por privacidade — veja `exemplo_base.xlsx` para o formato esperado.*

## Roadmap

- [ ] Pipeline Python de ingestão OFX (substituindo o N8N) com derivação das colunas de negócio (competência, data de pagamento, banco, forma de pagamento)
- [ ] Coluna Canal (delivery vs. presencial) como feature adicional
- [ ] Subcategorias (v2, quando houver volume por classe)
- [ ] Retreino automatizado com as correções acumuladas

---

*Projeto desenvolvido como capstone de um estudo estruturado de fundamentos de Machine Learning (10 dias), em preparação para a Formação em Ciência de Dados. A IA foi usada como ferramenta de execução e aprendizado; as decisões, os dados e a direção foram meus.*
