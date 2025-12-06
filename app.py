# app.py  — Versão ajustada para Streamlit (substitua o seu conteúdo atual por este)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.figure_factory as ff
import io
import numpy as np
import statsmodels.api as sm

# ------------------------------------------------------------
# FUNÇÃO DE LEITURA E LIMPEZA (alterada / reforçada)
# ------------------------------------------------------------
@st.cache_data
def fetch_and_clean_data():
    """
    Lê os CSVs remotos e garante que:
    - as colunas macro tenham nomes padronizados,
    - os retornos logaritmos de CÂMBIO e CDS sejam criados no dataframe final,
    - e retorna (df_parcial, df_final).
    """
    url_parcial = 'https://raw.githubusercontent.com/sanderpiva/fatores_macro_docs/main/resultados_modelo_json/parcial_merged_dfs_cds.csv'
    url_final = 'https://raw.githubusercontent.com/sanderpiva/fatores_macro_docs/main/resultados_modelo_json/final_merged_dfs_with_log_returns.csv'
    
    try:
        d_frame_parcial = pd.read_csv(url_parcial)
        d_frame_final = pd.read_csv(url_final)
        
        # ----------------------------
        #  NORMALIZAÇÃO / RENOMEAÇÃO
        # ----------------------------
        # Aqui padronizamos nomes que podem estar diferentes entre Colab e Streamlit.
        # Ajuste o rhs se seus nomes reais forem diferentes no CSV do Streamlit.
        d_frame_final = d_frame_final.rename(columns={
            # Exemplo de mapeamento comum — altere se necessário
            'Taxa de juros - Meta Selic definida pelo Copom - % a.a.': 'Taxa Selic - a.a.',
            'Taxa Selic a.a.': 'Taxa Selic - a.a.',
            'Taxa Cambio u.m.c./US$': 'Taxa Cambio u.m.c./US$',  # mantemos o nome padrão do Colab
            'CLOSE_CAMBIO': 'Taxa Cambio u.m.c./US$',
            'close_cambio': 'Taxa Cambio u.m.c./US$',
            'CDS': 'CDS',
            'CLOSE_CDS': 'CDS',
            'close_cds': 'CDS'
        })
        
        # ---------------------------------------------------------
        #  CRIAR OS RETORNOS LOGARÍTMICOS DE CÂMBIO E CDS (essencial)
        # ---------------------------------------------------------
        # Só criamos se as colunas brutas existirem; se não existirem,
        # deixamos como NaN — isso evita KeyError ao executar no Streamlit.
        if 'Taxa Cambio u.m.c./US$' in d_frame_final.columns:
            d_frame_final['RETORNO_LOG_CAMBIO'] = np.log(
                d_frame_final['Taxa Cambio u.m.c./US$'] /
                d_frame_final['Taxa Cambio u.m.c./US$'].shift(1)
            )
        else:
            # marca explicitamente a coluna ausente (ajuda no debug)
            d_frame_final['RETORNO_LOG_CAMBIO'] = np.nan

        if 'CDS' in d_frame_final.columns:
            d_frame_final['RETORNO_LOG_CDS'] = np.log(
                d_frame_final['CDS'] /
                d_frame_final['CDS'].shift(1)
            )
        else:
            d_frame_final['RETORNO_LOG_CDS'] = np.nan

        # Garantir que a coluna 'Data' seja do tipo datetime (opcional)
        if 'Data' in d_frame_final.columns:
            d_frame_final['Data'] = pd.to_datetime(d_frame_final['Data'], errors='coerce')

        return d_frame_parcial, d_frame_final
        
    except Exception as e:
        st.error(f"Erro ao carregar os dados. Verifique a URL ou o formato: {e}")
        # Retorna dataframes vazios para não quebrar a app
        return pd.DataFrame(), pd.DataFrame()


# ------------------------------------------------------------
# CARREGA (uma vez, com cache)
# ------------------------------------------------------------
df_parcial, df_final = fetch_and_clean_data()


# ------------------------------------------------------------
# FUNÇÃO DO MODELO (ajustada para verificação extra)
# ------------------------------------------------------------
def run_macro_model(df, target_cols):
    """
    Roda OLS para cada coluna em target_cols usando as variáveis:
    ['Taxa Selic - a.a.', 'RETORNO_LOG_CAMBIO', 'RETORNO_LOG_CDS'].
    A função verifica a existência das colunas e dá mensagens claras via Streamlit.
    """
    # Checagem inicial: dataframe vazio
    if df.empty:
        return {'Erro Geral': 'DataFrame final está vazio. Não é possível rodar o modelo.'}
    
    # Padroniza o nome da Selic dentro da função (caso não tenha sido renomeado)
    if 'Taxa Selic - a.a.' not in df.columns and 'Taxa Selic a.a.' in df.columns:
        df = df.rename(columns={'Taxa Selic a.a.': 'Taxa Selic - a.a.'})
    
    # Verifica presença das colunas brutas necessárias para gerar os retornos
    required_raw = ['Taxa Selic - a.a.', 'Taxa Cambio u.m.c./US$', 'CDS']
    missing_raw = [c for c in required_raw if c not in df.columns]
    
    # Se as colunas brutas existirem mas ainda não existirem os retornos, recalcula
    if 'RETORNO_LOG_CAMBIO' not in df.columns and 'Taxa Cambio u.m.c./US$' in df.columns:
        df['RETORNO_LOG_CAMBIO'] = np.log(df['Taxa Cambio u.m.c./US$'] / df['Taxa Cambio u.m.c./US$'].shift(1))
    if 'RETORNO_LOG_CDS' not in df.columns and 'CDS' in df.columns:
        df['RETORNO_LOG_CDS'] = np.log(df['CDS'] / df['CDS'].shift(1))
    
    # Prepare lista completa que vamos exigir no dropna
    subset_cols = ['Taxa Selic - a.a.', 'RETORNO_LOG_CAMBIO', 'RETORNO_LOG_CDS'] + target_cols
    
    # Verifica quais colunas deste subset NÃO existem e mostra mensagem amigável
    missing_cols = [c for c in subset_cols if c not in df.columns]
    if missing_cols:
        # Retornamos um dicionário com o erro para que a UI exiba isto claramente
        return {'Erro Colunas Faltantes': f"As seguintes colunas não existem no DataFrame: {missing_cols}. Verifique o pipeline de merges/renomeações."}
    
    # Dropna só depois de garantir que todas as colunas existem
    df_cleaned = df.dropna(subset=subset_cols)
    
    # X com constantes
    X = df_cleaned[['Taxa Selic - a.a.', 'RETORNO_LOG_CAMBIO', 'RETORNO_LOG_CDS']]
    X = sm.add_constant(X)
    
    results = {}
    for y_var in target_cols:
        if y_var not in df_cleaned.columns:
            results[y_var] = f"Coluna alvo {y_var} não encontrada no DataFrame."
            continue
        Y = df_cleaned[y_var]
        try:
            model = sm.OLS(Y, X, missing='drop').fit()
            results[y_var] = model.summary().as_text()
        except Exception as e:
            results[y_var] = f"Erro ao rodar o modelo OLS para {y_var}: {e}"
    return results


# ------------------------------------------------------------
#  BARRA LATERAL (mantive seu layout; só adicionei debug opcional)
# ------------------------------------------------------------
st.sidebar.header('Configurações', divider='blue')

data_expander = st.sidebar.expander(label="# **Dados Tabulares**", icon=":material/table:")
with data_expander:
    with st.form("settings_form", clear_on_submit=False):
        st.markdown("**Selecione as Visualizações**")
        explain_data = st.checkbox("Significado dos Dados", key="explain")
        data_in_table_parcial = st.checkbox("Exibir Tabela de Dados Parcial", key="table_parcial")
        data_in_table_final = st.checkbox("Exibir Tabela de Dados Final", key="table_final")
        data_info = st.checkbox("Informações dataframe final", key="info")
        data_described = st.checkbox("Resumir dados dataframe final (Describe)", key="describe")
        model_selic_cambio_cds = st.checkbox("Modelo Selic + Cambio + CDS", key="model_sc_cds")
        
        settings_form_submitted = st.form_submit_button("Carregar")

graph_expander = st.sidebar.expander("# **Gráficos**", icon=":material/monitoring:")
with graph_expander:
    with st.form("graphs_form", clear_on_submit=False):
        pass_per_class_graph = st.checkbox("Passageiros por Classe")
        age_hitogram = st.checkbox("Frequência de Idade")
        survived = st.checkbox("% de Sobreviventes")
        class_survived = st.checkbox("Sobreviventes por Classe")
        class_p_survived = st.checkbox("% de Sobreviventes por Classe")
        corr_class_survived = st.checkbox("Correlação Sobreviventes vs Classe")
        corr_1class3_survived = st.checkbox("Correlação Sobreviventes vs Classe (1 e 3)")
        sex_survived = st.checkbox("Sobreviventes por Sexo")
        corr_sex_survived = st.checkbox("Correlação Sobreviventes vs Sexo")
        
        graphs_form_submitted = st.form_submit_button("Gerar")

# Página Principal
st.header('Projeto Fatores Macroeconomicos', divider='blue')

data_meaning = '''
- `Variável`: Significado
- `Data`: Data de referencia que relaciona os dados (06 out 2020 - 01 out 2025)
- `Taxa Selic - a.a.`: Taxa Selic em porcentual ao ano
- `Taxa Cambio u.m.c./US$`: Taxa de câmbio unidade monetária corrente/US$
- `CDS`: Risco Brasil CDS
- `Itau`: Preco da ação (fechamento) do Itaú
- `RETORNO_LOG_Itau`: Calculo retorno logaritmo para o preço da ação da Itau
- `Petrobras`: Preco da ação (fechamento) da Petrobras
- `RETORNO_LOG_Petrobras`: Calculo retorno logaritmo para o preço da ação da Petrobras
- `Vale do Rio Doce`: Preco da ação (fechamento) da Vale Rio Doce
- `RETORNO_LOG_Vale Rio Doce`: Calculo retorno logaritmo para o preço da ação da Vale
'''

# Quando o form for submetido
if settings_form_submitted:
    if explain_data:
        st.subheader("Dicionário dos Dados", divider="gray")
        st.markdown(data_meaning)
    
    if data_in_table_parcial:
        st.subheader("Tabela de Dados Parcial", divider="gray")
        st.write(df_parcial)
    
    if data_in_table_final:
        st.subheader("Tabela de Dados Final", divider="gray")
        st.write(df_final)
    
    if data_described:
        st.subheader("Resumo dos dados: dataframe final", divider="gray")
        st.write(df_final.describe())
    
    if data_info:
        st.subheader("Informação dos dados: dataframe Final", divider="gray")
        try:
            if 'Data' in df_final.columns:
                df_final['Data'] = pd.to_datetime(df_final['Data'], errors='coerce')
            else:
                st.warning("Aviso: A coluna 'Data' não foi encontrada no DataFrame. Verifique o nome da coluna.")
        except KeyError:
            st.warning("Aviso: Erro ao converter Data para datetime.")

        buffer_captura = io.StringIO()
        df_final.info(buf=buffer_captura)
        st.code(buffer_captura.getvalue(), language='text')

    if model_selic_cambio_cds:
        st.subheader("Modelo Selic + Cambio + CDSx", divider="gray")
        
        # Ações alvo
        acoes_retorno = ['RETORNO_LOG_Itau', 'RETORNO_LOG_Petrobras', 'RETORNO_LOG_Vale do Rio Doce']
        
        # Executa modelo
        model_results = run_macro_model(df_final.copy(), acoes_retorno)
        
        # Se houver erro de colunas faltantes, exibimos de forma clara
        if isinstance(model_results, dict) and 'Erro Colunas Faltantes' in model_results:
            st.error(model_results['Erro Colunas Faltantes'])
        elif isinstance(model_results, dict) and 'Erro Geral' in model_results:
            st.error(model_results['Erro Geral'])
        else:
            # Exibe os resultados
            for acao, summary_text in model_results.items():
                st.markdown(f"### Resultados da Regressão para: **{acao}**")
                st.code(summary_text, language='text')

                # Extrair R-squared (melhora: regex mais robusto)
                import re
                r_sq_match = re.search(r'R-squared:\s+([0-9]*\.[0-9]+)', summary_text)
                if r_sq_match:
                    r_squared = float(r_sq_match.group(1))
                    st.info(f"O **$R^2$** para **{acao}** é de **{r_squared:.4f}**.")
                    st.caption("Este valor indica a porcentagem da variação no Retorno da Ação que é explicada pelas variáveis macroeconômicas (Selic, Câmbio e CDS).")
##

# Ao submeter o form de gráficos
if graphs_form_submitted:
    if pass_per_class_graph:
        st.subheader("Passageiros por Classe", divider="gray")
        
        st.bar_chart(data=df['class'].value_counts().sort_index(), x_label="Classe", y_label="Nº Passageiros", color="#8A2BE2")

if graphs_form_submitted:
    if age_hitogram:
        st.subheader("Histograma por Idade", divider="gray")
        
        # Agrupa os dados
        data = [df['age']]
        labels = ['Idade']
        # Cria o distplot com bin_size customizado
        fig = ff.create_distplot(data, labels, bin_size=[2], colors=["#8A2BE2"])

        # Plot!
        st.plotly_chart(fig, use_container_width=True)

if graphs_form_submitted:
    if survived:
        st.subheader("% de Sobreviventes", divider="gray")

        frame = df['survived'].value_counts().sort_index().to_frame("count")
        frame = frame.reset_index(names="survived")
        frame.replace(0, 'Morreu', inplace=True)
        frame.replace(1, 'Sobreviveu', inplace=True)
        
        # Gráfico
        plt.figure(figsize=(12, 9))
        plt.pie(frame['count'], labels=frame['survived'], autopct='%1.0f%%')
        st.pyplot(plt)

if graphs_form_submitted:
    if class_survived:
        st.subheader("Sobreviventes por Classe", divider="gray")
        
        df_ordered = df.sort_values('class', ascending=True)

        plt.figure(figsize=(8, 6))
        graph = sns.countplot(x='class', hue='survived', data=df_ordered)

        legends, _ = graph.get_legend_handles_labels()
        graph.legend(legends, ['Morreu','Sobreviveu'], title='Status do Passageiro(a)')

        plt.xlabel('Classe')
        plt.ylabel('Número de Passageiros')

        graph.bar_label(graph.containers[0])
        graph.bar_label(graph.containers[1])
        st.pyplot(plt)

if graphs_form_submitted:
    if class_p_survived:
        st.subheader("% de Sobreviventes por Classe", divider="gray")

        df_ordered = df.sort_values('class', ascending=True)
        
        # Percentual de sobreviventes
        survived_percent = df_ordered['survived'].mean() * 100

        # Gráfico
        plt.figure(figsize=(8, 6))

        mean = df_ordered.groupby('class')['survived'].mean() * 100
        mean['Mean'] = survived_percent
        df_mean = mean.to_frame().reset_index()

        colors = ['red' if x < survived_percent else 'yellow' for x in df_mean['survived']]

        graph = sns.barplot(x='class', y="survived", hue='class', data=df_mean.round(), palette=colors)

        plt.xlabel('Classe')
        plt.ylabel('% de sobreviventes')

        graph.bar_label(graph.containers[0])
        graph.bar_label(graph.containers[1])
        graph.bar_label(graph.containers[2])
        graph.bar_label(graph.containers[3])
        st.pyplot(plt)

if graphs_form_submitted:
    if corr_class_survived:
        st.subheader("Correlação Classe vs Status Sobrevivência", divider="gray")
        
        df_ordered = df.sort_values('class', ascending=True)
        # Geral
        frame = df_ordered

        # Correlação Classe vs Sobrevivente
        correlation = frame[['survived', 'pclass']].corr()

        # Gráfico
        plt.figure(figsize=(8, 6))
        heatmap = sns.heatmap(correlation, vmin=-1, vmax=1, annot=True)
        st.pyplot(plt)

if graphs_form_submitted:
    if corr_1class3_survived:
        st.subheader("Correlação Classe (Rico vs Pobre) vs Status Sobrevivência", divider="gray")
        
        df_ordered = df.sort_values('class', ascending=True)
        df_ordered_removed_class2 = df_ordered.drop(df_ordered[df_ordered['pclass'] == 2].index)
        
        # Geral
        frame = df_ordered_removed_class2

        st.write(frame)
        # Correlação Classe vs Sobrevivente
        correlation = frame[['survived', 'pclass']].corr()

        # Gráfico
        plt.figure(figsize=(8, 6))
        heatmap = sns.heatmap(correlation, vmin=-1, vmax=1, annot=True)
        st.pyplot(plt)


if graphs_form_submitted:
    if sex_survived:
        st.subheader("Sobreviventes por Sexo", divider="gray")
        data = pd.DataFrame(df.groupby('sex')['survived'].value_counts().sort_index())
        data = data.reset_index();
        
        chart_data = pd.DataFrame(
            {
                "Sexo": list(data['sex']),
                "Pessoas": list(data['count']),
                "Situação": list(["Vivo" if x == 1 else "Morto" for x in data['survived']])
            }
        )
        
        st.bar_chart(chart_data, x="Sexo", y="Pessoas", color="Situação")

if graphs_form_submitted:
    if corr_sex_survived:
        st.subheader("Correlação Sexo vs Status Sobrevivência", divider="gray")

        # Geral
        df_ordered = df.sort_values('class', ascending=True)
        frame = df_ordered
        frame.replace('male', 0, inplace=True)
        frame.replace('female', 1, inplace=True)

        # Correlação Sexo vs Sobrevivente
        correlation = frame[['survived', 'sex']].corr()
        
        # Gráfico
        plt.figure(figsize=(8, 6))
        heatmap = sns.heatmap(correlation, vmin=-1, vmax=1, annot=True)
        st.pyplot(plt)