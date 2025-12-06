import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.figure_factory as ff
import io
import numpy as np
import statsmodels.api as sm

@st.cache_data
@st.cache_data
def fetch_and_clean_data():
    url_parcial = 'https://raw.githubusercontent.com/sanderpiva/fatores_macro_docs/main/resultados_modelo_json/parcial_merged_dfs_cds.csv'
    # ATUALIZE ESTA URL para o link do novo CSV que você subiu, que já está completo
    url_final = 'https://raw.githubusercontent.com/sanderpiva/fatores_macro_docs/main/resultados_modelo_json/df_final_model.csv' 
    
    try:
        d_frame_parcial = pd.read_csv(url_parcial)
        d_frame_final = pd.read_csv(url_final)
        
        # --- REMOVIDA TODA A LÓGICA DE CÁLCULO E ARREDONDAMENTO ---
        # O Streamlit apenas carrega os dados já prontos do CSV
        
        return d_frame_parcial, d_frame_final
        
    except Exception as e:
        # Este bloco agora captura apenas falhas de carregamento de URL/Rede
        st.error(f"Erro ao carregar os dados. Verifique a URL ou o formato: {e}")
        return pd.DataFrame(), pd.DataFrame()
# Carrega os DataFrames (apenas uma vez, graças ao cache)
df_parcial, df_final = fetch_and_clean_data()


def run_macro_model(df, target_cols):
    """
    Roda a Regressão OLS com método de diagnóstico robusto para KeyError.
    """
    # 0. Verificação de Dados (Guardrail para DF vazio)
    if df.empty:
        return {'Erro Geral': 'DataFrame final está vazio. Não é possível rodar o modelo.'}
    
    # Colunas necessárias para o modelo (X e Y)
    required_macro_cols = ['Taxa Selic a.a.', 'RETORNO_LOG_CAMBIO', 'RETORNO_LOG_CDS']
    required_cols = required_macro_cols + target_cols
    
    # NOVO MÉTODO: Verificação explícita de cada coluna ausente
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        # Se houver colunas faltando, retorna o erro com a lista de colunas encontradas
        error_message = f"""
        Erro: O modelo OLS não pode ser executado.
        
        As seguintes colunas estão faltando: 
        {", ".join(missing_cols)}
        
        Colunas disponíveis no DataFrame (Verifique espaços, maiúsculas e minúsculas):
        {df.columns.tolist()}
        """
        # Usamos st.error (assumindo que esta parte da função rodará no bloco settings_form_submitted)
        return {'KeyError Isolado - Dados Faltando': error_message}

    # Se chegamos aqui, as colunas existem e podemos prosseguir
    
    # 1. Limpeza de Dados: usamos required_cols, que agora sabemos que existem
    df_cleaned = df.dropna(subset=required_cols)
    
    # 2. Definição das Variáveis Preditoras (X)
    X = df_cleaned[required_macro_cols]
    X = sm.add_constant(X) 
    
    results = {}
    
    # 3. Rodar e Armazenar os Modelos (Loop OLS)
    for y_var in target_cols:
        Y = df_cleaned[y_var]
        try:
            model = sm.OLS(Y, X, missing='drop').fit()
            results[y_var] = model.summary().as_text()
            
        except ValueError as e:
            results[y_var] = f"Erro ao rodar o modelo OLS para {y_var}: {e}"
            
    return results
#
# --- 2. BARRA LATERAL (Seu Código Adaptado) ---
st.sidebar.header('Configurações', divider='blue')

data_expander = st.sidebar.expander(label="# **Dados Tabulares**", icon=":material/table:")
with data_expander:
    # O form é crucial para agrupar as ações de filtro e só atualizar a tela quando o botão for pressionado
    with st.form("settings_form", clear_on_submit=False):
        st.markdown("**Selecione as Visualizações**")
        explain_data = st.checkbox("Significado dos Dados", key="explain")
        data_in_table_parcial = st.checkbox("Exibir Tabela de Dados Parcial", key="table_parcial")
        data_in_table_final = st.checkbox("Exibir Tabela de Dados Final", key="table_final")
        data_info = st.checkbox("Informações dataframe final", key="info")
        data_described = st.checkbox("Resumir dados dataframe final (Describe)", key="describe")
        model_selic_cambio_cds = st.checkbox("Modelo Selic + Cambio + CDS", key="model_sc_cds")
        
        # O botão de submissão é necessário para que as checagens acima sejam processadas
        settings_form_submitted = st.form_submit_button("Carregar")

#

graph_expander = st.sidebar.expander("# **Gráficos**", icon=":material/monitoring:")
# st.sidebar.subheader('Gráficos')
with graph_expander:
    # Formulário dos gráficos
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

# === Página Principal ===
st.header('Projeto Fatores Macroeconomicos', divider='blue')

# Um markdown de múltiplas linhas
data_meaning = '''

- `Variável`: Significado

- `Data`: Data de referencia que relaciona os dados (06 out 2020 - 01 out 2025)
- `Taxa Selic a.a.`: Taxa Selic em porcentual ao ano
- `Taxa Cambio u.m.c./US$`: Taxa de câmbio unidade monetária corrente/US$
- `CDS`: Risco Brasil CDS
- `Itau`: Preco da ação (fechamento) do Itaú
- `RETORNO_LOG_Itau`: Calculo retorno logaritmo para o preço da ação da Itau
- `Petrobras`: Preco da ação (fechamento) da Petrobras
- `RETORNO_LOG_Petrobras`: Calculo retorno logaritmo para o preço da ação da Petrobras
- `Vale do Rio Doce`: Preco da ação (fechamento) da Vale Rio Doce
- `RETORNO_LOG_Vale Rio Doce`: Calculo retorno logaritmo para o preço da ação da Vale
'''

# Ao submeter o form de dados tabulares
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
        #Garantindo a conversao da data em datatime
        try:
            df_final['Data'] = pd.to_datetime(df_final['Data'], errors='coerce')
        except KeyError:
            st.warning("Aviso: A coluna 'Data' não foi encontrada no DataFrame. Verifique o nome da coluna.")

        # Cria um objeto para capturar a saída de texto
        buffer_captura = io.StringIO()
        
        # Redireciona a saída impressa de .info() para o nosso buffer
        df_final.info(buf=buffer_captura)
        
        # Exibe o conteúdo capturado na webapp Streamlit
        st.code(buffer_captura.getvalue(), language='text')
    
    if model_selic_cambio_cds:
        st.subheader("Modelo Selic + Cambio + CDSx", divider="gray")
        
        # Variáveis dependentes para as 3 ações
        acoes_retorno = ['RETORNO_LOG_Itau', 'RETORNO_LOG_Petrobras', 'RETORNO_LOG_Vale do Rio Doce']
        
        # Executa o modelo
        model_results = run_macro_model(df_final.copy(), acoes_retorno)
        
        # Exibe os resultados
        for acao, summary_text in model_results.items():
            st.markdown(f"### Resultados da Regressão para: **{acao}**")
            # st.code é ideal para exibir o summary formatado
            st.code(summary_text, language='text')

            # Você pode adicionar um st.write para interpretar o R-Quadrado [cite: 46]
            import re
            r_sq_match = re.search(r'R-squared:\s+(\d\.\d+)', summary_text)
            
            if r_sq_match:
                r_squared = float(r_sq_match.group(1))
                st.info(f"O **$R^2$ (Coeficiente de Determinação)** para **{acao}** é de **{r_squared:.4f}**.")
                st.caption("Este valor indica a porcentagem da variação no Retorno da Ação que é explicada pelas variáveis macroeconômicas (Selic, Câmbio e CDS)[cite: 46].")

#


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