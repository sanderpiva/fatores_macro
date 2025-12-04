import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.figure_factory as ff

@st.cache_data
def fetch_and_clean_data():
    url_parcial = 'https://raw.githubusercontent.com/sanderpiva/fatores_macro_docs/main/resultados_modelo_json/parcial_merged_dfs_cds.csv'
    url_final = 'https://raw.githubusercontent.com/sanderpiva/fatores_macro_docs/main/resultados_modelo_json/final_merged_dfs_with_log_returns.csv'
    
    try:
        d_frame_parcial = pd.read_csv(url_parcial)
        d_frame_final = pd.read_csv(url_final)
        return d_frame_parcial, d_frame_final
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {e}")
        # Retorna DataFrames vazios em caso de erro para não quebrar o resto do app
        return pd.DataFrame(), pd.DataFrame()

# Carrega os DataFrames (apenas uma vez, graças ao cache)
df_parcial, df_final = fetch_and_clean_data()

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
        data_described = st.checkbox("Resumir Dados (Describe)", key="describe")
        
        # O botão de submissão é necessário para que as checagens acima sejam processadas
        settings_form_submitted = st.form_submit_button("Carregar")

# --- 3. EXIBIÇÃO CONDICIONAL NO CORPO PRINCIPAL ---
st.title("📊 Dashboard de Análise de Fatores Macro")

# 3.1. Exibição do Significado dos Dados (Exemplo)
if explain_data:
    st.header("📚 Significado dos Dados")
    st.info("Aqui você colocaria a documentação ou metadados de suas colunas e DataFrames.")
    st.markdown("- **DataFrame Parcial:** Contém dados de Cédulas de Crédito e Variações... ")
    st.markdown("- **DataFrame Final:** Inclui retornos logarítmicos e fatores macro consolidados... ")
    st.markdown("---")


# 3.2. Exibição da Tabela de Dados Parcial
if data_in_table_parcial:
    st.header("📋 Tabela de Dados Parcial")
    # Usa st.dataframe para exibir a tabela
    st.dataframe(df_parcial)
    
    if data_described:
        st.subheader("Resumo Estatístico (Parcial)")
        st.dataframe(df_parcial.describe(include='all'))
    st.markdown("---")


# 3.3. Exibição da Tabela de Dados Final
if data_in_table_final:
    st.header("📋 Tabela de Dados Final")
    # Usa st.dataframe para exibir a tabela
    st.dataframe(df_final)

    if data_described:
        st.subheader("Resumo Estatístico (Final)")
        st.dataframe(df_final.describe(include='all'))
    st.markdown("---")
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
st.header('Projeto X', divider='blue')

# Um markdown de múltiplas linhas
data_meaning = '''

- `Variável`: Significado

- `Survived`: Se o passageiro sobrevieu (0 = não, 1 = sim)
- `Pclass`: Classe (1 = primeira classe, 2 = segunda classe, 3 = terceira classe)
- `Sex`: Gênero
- `Age`: Idade do Passageiro
- `SibSp`: Número de irmãos/cônjuges a bordo
- `Parch`: Número de pais/crianças a bordo
- `Fare`: Tarifa paga pelo bilhete
- `Embarked`: Porto de embarque (C = Cherbourg, Q = Queenstown, S = Southampton)
- `Class`: Equivalente a `Pclass` (1 = 1ª classe, 2 = 2ª classe, 3 = 3ª classe)
- `Who`: Categoria do passageiro (homem, mulher, criança)
- `Adult_male`: Se o passageiro é um homem adulto ou não (Verdadeiro ou Falso)
- `Deck`: Convés da cabine
- `Embark_town`: Porto de embarque (Cherbourg, Queenstown, Southampton)
- `Alive`: Status de sobrevivência (sim ou não)
- `Alone`: Se o passageiro está sozinho ou não (Verdadeiro ou Falso)
'''

# Ao submeter o form de dados tabulares
if settings_form_submitted:
    if explain_data:
        st.subheader("Dicionário dos Dados", divider="gray")
        st.markdown(data_meaning)
    
    if data_in_table:
        st.subheader("Tabela da Dados", divider="gray")
        st.write(df)
    
    if data_described:
        st.subheader("Resumo dos Dados", divider="gray")
        st.write(df.describe())

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