import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.figure_factory as ff
import io
import re
import numpy as np
import statsmodels.api as sm

@st.cache_data
def fetch_and_clean_data():
    url_parcial = 'https://raw.githubusercontent.com/sanderpiva/fatores_macro_docs/main/resultados_modelo_json/parcial_merged_dfs_cds.csv'
    url_final = 'https://raw.githubusercontent.com/sanderpiva/fatores_macro_docs/main/resultados_modelo_json/df_final_model.csv' 
    
    try:
        d_frame_parcial = pd.read_csv(url_parcial)
        d_frame_final = pd.read_csv(url_final)
        
        return d_frame_parcial, d_frame_final
        
    except Exception as e:
        st.error(f"Erro ao carregar os dados. Verifique a URL ou o formato: {e}")
        return pd.DataFrame(), pd.DataFrame()

df_parcial, df_final = fetch_and_clean_data()

#
def run_macro_model(df, target_cols, X_cols):
    """
    Roda a Regressão OLS de forma robusta, aceitando uma lista dinâmica de variáveis preditoras (X_cols).
    """
    if df.empty:
        return {'Erro Geral': 'DataFrame final está vazio. Não é possível rodar o modelo.'}
    
    required_macro_cols = X_cols
    required_cols = required_macro_cols + target_cols
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        error_message = f"""
        Erro: O modelo OLS não pode ser executado com as variáveis {X_cols}.
        
        As seguintes colunas estão faltando (Verifique a URL do CSV e o nome exato): 
        {", ".join(missing_cols)}
        
        Colunas disponíveis no DataFrame:
        {df.columns.tolist()}
        """
        return {'KeyError Isolado - Dados Faltando': error_message}
   
    df_cleaned = df.dropna(subset=required_cols)
    
    X = df_cleaned[required_macro_cols]
    X = sm.add_constant(X) 
    
    results = {}
    
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
        data_in_table_final = st.checkbox("Exibir Tabela de Dados final", key="table_final")
        data_info = st.checkbox("Informações dataframe final", key="info")
        data_described = st.checkbox("Resumir dados dataframe final (Describe)", key="describe")
        model_selic_cambio_cds = st.checkbox("Modelo Selic + Cambio + CDS", key="model_sc_cds")
        model_cambio_cds = st.checkbox("Modelo Câmbio + CDS", key="model_c_cds")
        
        # O botão de submissão é necessário para que as checagens acima sejam processadas
        settings_form_submitted = st.form_submit_button("Carregar")

#

graph_expander = st.sidebar.expander("# **Gráficos**", icon=":material/monitoring:")
# st.sidebar.subheader('Gráficos')
with graph_expander:
    # Formulário dos gráficos
    with st.form("graphs_form", clear_on_submit=False):
        
        grap_log_return_cambio_cds = st.checkbox("Gráficos de Série Temporal do Retorno logaritmo da Taxa de Cambio e Risco Brasil - CDS")
        grap_log_return_stock_prices = st.checkbox("Gráficos de Série Temporal do Retorno logaritmo dos Preços das Ações")
        corr_variables = st.checkbox("Correlação Variáveis Independentes vs Variáveis Dependentes")
        
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

# Variáveis dependentes (Y) para as 3 ações

acoes_retorno = ['RETORNO_LOG_Itau', 'RETORNO_LOG_Petrobras', 'RETORNO_LOG_Vale Rio Doce'] 

# Variáveis preditoras (X)
X_macro_sc_cds = ['Taxa Selic - a.a.', 'RETORNO_LOG_CAMBIO', 'RETORNO_LOG_CDS'] # Modelo 1
X_macro_c_cds = ['RETORNO_LOG_CAMBIO', 'RETORNO_LOG_CDS'] # Modelo Final

# Ao submeter o form de dados tabulares
if settings_form_submitted:
    if explain_data:
        st.subheader("Dicionário dos Dados", divider="gray")
        st.markdown(data_meaning)
    
    if data_in_table_parcial:
        st.subheader("Tabela de Dados Parcial", divider="gray")
        st.write(df_parcial)
    
    if data_in_table_final:
        st.subheader("Tabela de Dados Final (Modelo)", divider="gray")
        st.write(df_final)
    
    if data_info:
        st.subheader("Informações sobre os dados: dataframe final (Modelo)", divider="gray")
        try:
            df_final['Data'] = pd.to_datetime(df_final['Data'], errors='coerce')
        except KeyError:
            st.warning("Aviso: A coluna 'Data' não foi encontrada no DataFrame. Verifique o nome da coluna.")

        buffer_captura = io.StringIO()
        
        df_final.info(buf=buffer_captura)
        
        st.code(buffer_captura.getvalue(), language='text')
    
    if data_described:
        st.subheader("Resumo dos dados: dataframe final (Modelo)", divider="gray")
        st.write(df_final.describe())
    
    if model_selic_cambio_cds:
        st.subheader("Modelo Selic + Câmbio + CDS", divider="gray") 
    
        model_results = run_macro_model(df_final.copy(), acoes_retorno, X_macro_sc_cds)
        
        for acao, summary_text in model_results.items():
            st.markdown(f"### Resultados da Regressão para: **{acao}**")
            st.code(summary_text, language='text')

            r_sq_match = re.search(r'R-squared:\s+(\d\.\d+)', summary_text)
            
            if r_sq_match:
                r_squared = float(r_sq_match.group(1))
                st.info(f"O **$R^2$ (Coeficiente de Determinação)** para **{acao}** é de **{r_squared:.4f}**.")
                st.caption("Este valor indica a porcentagem da variação no Retorno da Ação que é explicada pelas variáveis macroeconômicas (Selic, Câmbio e CDS).")
        
        st.write("Esse modelo foi descartado, pois a regra p < 0.05 não é atendida em todos os casos. Exemplo: Para a Vale do Rio Doce, a Taxa Selic a.a. teve p > 0.05")

    if model_cambio_cds:
        st.subheader("Modelo Câmbio + CDS (Adotado)", divider="gray")
        
        model_results = run_macro_model(df_final.copy(), acoes_retorno, X_macro_c_cds)
    
        for acao, summary_text in model_results.items():
            st.markdown(f"### Resultados da Regressão para: **{acao}**")
            st.code(summary_text, language='text')

            
            r_sq_match = re.search(r'R-squared:\s+(\d\.\d+)', summary_text)
            
            if r_sq_match:
                r_squared = float(r_sq_match.group(1))
                st.info(f"O **$R^2$ (Coeficiente de Determinação)** para **{acao}** é de **{r_squared:.4f}**.")
                st.caption("Este valor indica a porcentagem da variação no Retorno da Ação que é explicada pelas variáveis macroeconômicas (Câmbio e CDS).")
        
        
        st.info(f"**VANTAGENS do Modelo (Cambio + CDS) adotado:**")
        st.write("- Oferece o melhor equilíbrio entre simplicidade e capacidade de explicar a variação nos retornos das ações;")
        st.write("- Ambas as variáveis, (RETORNO_LOG_CAMBIO e RETORNO_LOG_CDS são estatisticamente significativas (p < 0.05) para os três ativos;")
        st.write("- Multicolinearidade aceitável: A multicolinearidade (Cond. No.  ≈121 ) entre Câmbio e CDS indica que eles se movem juntos, dificultando a interpretação isolada do efeito de cada um, porém não a capacidade preditiva do modelo como um todo;")
        st.write("- As tentativas com a combinação Selic + CDS, ou ainda, apenas o CDS não trouxeram poder explicativo relevante.")
#

# Ao submeter o form de gráficos

if graphs_form_submitted:
    
    
    if grap_log_return_cambio_cds:
        st.subheader("Gráficos de Série Temporal do Retorno logaritmo Taxa Cambio e Risco Brasil - CDS")

        df_usado = df_final.copy() 

        df_usado['Data'] = pd.to_datetime(df_usado['Data'], errors='coerce')

        acoes_retornos = ['RETORNO_LOG_CAMBIO', 'RETORNO_LOG_CDS']

        fig, axes = plt.subplots(3, 1, figsize=(15, 18), sharex=False) 

        for i, acao in enumerate(acoes_retornos):
        
            sns.lineplot(ax=axes[i], x='Data', y=acao, data=df_usado, color=plt.cm.viridis(i/len(acoes_retornos)))
            
            axes[i].set_title(f'Série Temporal do Retorno Logarítmico: {acao}', fontsize=16)
            axes[i].set_ylabel('Retorno Logarítmico', fontsize=12)
            axes[i].tick_params(axis='y', labelsize=10)
            axes[i].grid(True, linestyle='--', alpha=0.7)
            
            axes[i].tick_params(axis='x', rotation=45, labelsize=10)
            
            axes[i].set_xlabel('Data', fontsize=12) 

            plt.tight_layout()

            st.pyplot(fig)
    
    
    if grap_log_return_stock_prices:
        st.subheader("Gráficos de Série Temporal do Retorno logaritmo dos Preços das Ações", divider="gray")

        df_usado = df_final.copy() 

        df_usado['Data'] = pd.to_datetime(df_usado['Data'], errors='coerce')

        acoes_retornos = ['RETORNO_LOG_Itau', 'RETORNO_LOG_Petrobras', 'RETORNO_LOG_Vale Rio Doce']

        fig, axes = plt.subplots(3, 1, figsize=(15, 18), sharex=False) 

        for i, acao in enumerate(acoes_retornos):
        
            sns.lineplot(ax=axes[i], x='Data', y=acao, data=df_usado, color=plt.cm.viridis(i/len(acoes_retornos)))
            
            axes[i].set_title(f'Série Temporal do Retorno Logarítmico: {acao}', fontsize=16)
            axes[i].set_ylabel('Retorno Logarítmico', fontsize=12)
            axes[i].tick_params(axis='y', labelsize=10)
            axes[i].grid(True, linestyle='--', alpha=0.7)
            
            axes[i].tick_params(axis='x', rotation=45, labelsize=10)
            
            axes[i].set_xlabel('Data', fontsize=12) 


        plt.tight_layout()

        st.pyplot(fig)

if graphs_form_submitted:
    if corr_variables:
        st.subheader("Correlação Variáveis Independentes vs Variáveis Dependentes", divider="gray")

        correlation_matrix = df_final[['Taxa Selic - a.a.', 'RETORNO_LOG_CAMBIO', 'RETORNO_LOG_CDS', 'RETORNO_LOG_Itau', 'RETORNO_LOG_Petrobras', 'RETORNO_LOG_Vale Rio Doce']].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))

        sns.heatmap(
            correlation_matrix, 
            annot=True, 
            cmap='coolwarm', 
            fmt=".2f", 
            linewidths=.5, 
            ax=ax 
        )
        
        ax.set_title('Mapa de Calor da Correlação das Variáveis Independentes x Variáveis Dependentes', fontsize=16)
        
        st.pyplot(fig)
        st.info(f"**IMPORTANTE:**")
        st.write("Variáveis independentes: Taxa Selic - a.a., Retorno Logaritmo Cambio e Retorno Logaritmo CDS")
        st.write("Variáveis dependentes: Retorno Logaritmo Itau, Retorno Logaritmo Petrobras, Retorno Logaritmo Vale Rio Doce (Retorno Logaritmo das ações das empresas)")
        st.info(f"**COMPORTAMENTO?**")
        st.write("A medida que aumento o valor do cambio e/ou do risco CDS, o retorno logaritmo do preço das ações das empresas Itau, Petrobrás e Vale do Rio Doce tendem a diminuir.")
        st.write("Variável independente Taxa Selic - a.a. foi desconsiderada no modelo preditivo adotado")