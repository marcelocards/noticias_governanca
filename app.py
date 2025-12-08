import os
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np

# ==============================
# Configurações Padrão
# ==============================
TITLE_WEIGHT = 0.7
BODY_WEIGHT = 0.3
USE_ARTICLE_BODY = False 
MAX_BODY_CHARS = 4000

# ==============================
# Imports do core
# ==============================
from core import (
    get_stock_data, get_brent_prices, get_dollar_rate,
    get_news, process_news_with_sentiment, 
    daily_sentiment_series, build_daily_table, build_variations_table,
    CATEGORIES, calc_variations, corr_table
)

# ==============================
# Config da página e Estilos
# ==============================
st.set_page_config(
    page_title="Petrobras – Mercado, Notícias e Análise de Sentimento",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
.main-header { font-size: 2.1rem; margin-bottom: .25rem; }
.card { background: #fff; padding: .9rem; border-radius: 12px; border: 1px solid #eee; }
.badge { display:inline-block; padding:.18rem .5rem; border-radius:12px; font-size:.75rem; margin-right:.4rem; font-weight:600; text-transform: uppercase; }
.positivo { color:#065f46; background:#d1fae5; } 
.negativo { color:#991b1b; background:#fee2e2; } 
.neutro { color:#92400e; background:#fff7ed; }
</style>
""", unsafe_allow_html=True)


# ==============================
# Função de Carregamento de Dados (Cache)
# ==============================
@st.cache_data(show_spinner="Buscando dados de mercado e notícias...")
def load_data(
    end_date: datetime,
    days_ago: int,
    query_theme: str,
    use_body: bool = USE_ARTICLE_BODY, 
    w_title: float = TITLE_WEIGHT,
    w_body: float = BODY_WEIGHT,
) -> tuple:
    """
    Busca e processa dados de mercado, notícias e sentimento.
    """
    end_dt = datetime.combine(end_date, datetime.min.time())
    start_dt = end_dt - timedelta(days=days_ago)

    # 1. Dados de Mercado
    # Incluindo PBR e PBRA
    tickers = ["PETR3.SA", "PETR4.SA", "^BVSP", "PBR", "PBRA"] 
    stock_data = get_stock_data(tickers, start_dt, end_dt)
    brent_df = get_brent_prices(start_dt, end_dt)
    dollar_df = get_dollar_rate(start_dt, end_dt)

    # 2. Notícias
    period_str = f"{days_ago}d"
    query = "PETROBRAS" 
    news_raw = get_news(query=query, period=period_str, max_results=50, theme=query_theme)
    
    # 3. Processamento e Sentimento
    news_processed = process_news_with_sentiment(
        news_raw, use_body=use_body, w_title=w_title, w_body=w_body
    )

    # 4. Construção das Séries Diárias
    daily_sent = daily_sentiment_series(news_processed)
    daily_table = build_daily_table(stock_data, brent_df, dollar_df, daily_sent)

    return stock_data, brent_df, dollar_df, daily_table, news_processed

# ==============================
# Interface do Usuário (UI)
# ==============================

st.title("📈 Petrobras – Mercado, Notícias e Análise de Sentimento")

# --- Barra Lateral (Sidebar) ---
st.sidebar.header("⚙️ Configurações da Análise")

# Seletor de Período
today = datetime.now().date() 

days_ago = st.sidebar.slider(
    "Período de Análise (Dias)",
    min_value=7,
    max_value=180,
    value=30,
    step=7,
)
end_date = st.sidebar.date_input("Data Final", value=today, max_value=today)

# Seletor de Tema
theme_options = ["Todos"] + list(CATEGORIES.keys())
selected_theme = st.sidebar.selectbox(
    "Filtrar Notícias por Tema",
    options=theme_options,
    index=0,
)

# Configurações Avançadas
st.sidebar.subheader("Ajustes de Sentimento")

use_body_input = st.sidebar.checkbox(
    "Incluir corpo da notícia na análise (Lento)", 
    value=USE_ARTICLE_BODY,
    help="Se marcado, a aplicação tentará extrair o corpo do artigo (Newspaper3k), o que pode aumentar o tempo de carregamento."
)

title_weight_input = st.sidebar.slider(
    "Peso do Título (vs. Corpo)",
    min_value=0.0,
    max_value=1.0,
    value=TITLE_WEIGHT,
    step=0.1,
    help="Define a importância do score de sentimento calculado no título."
)
body_weight_input = 1.0 - title_weight_input
st.sidebar.markdown(f"*(Peso do Corpo: **{body_weight_input:.1f}**)*")

# Execução da Função de Carregamento de Dados
stock_data, brent_df, dollar_df, daily_table, news_processed = load_data(
    end_date=end_date,
    days_ago=days_ago,
    query_theme=selected_theme,
    use_body=use_body_input, 
    w_title=title_weight_input, 
    w_body=body_weight_input, 
)

if daily_table.empty and not news_processed:
    st.error("Não foi possível carregar dados de mercado e/ou notícias para o período e configurações selecionados. Tente reduzir o período.")
    st.stop()

# --- Abas de Visualização ---
tab1, tab2, tab3 = st.tabs(["Dashboard de Mercado e Sentimento", "Análise de Notícias Detalhada", "Correlação"])

with tab1:
    st.markdown("## 📊 Dashboard de Mercado e Sentimento")
    st.write(f"Análise dos últimos **{days_ago} dias** (até **{end_date.strftime('%d/%m/%Y')}**).")

    # 1. Gráfico de Preços Normalizados e Sentimento Diário
    st.subheader("Preços Normalizados (Índice 100)")
    
    # Normalizar as colunas de preços para o índice 100
    normalized_prices = {}
    for col in ["IBOV", "PETR4", "PETR3", "Brent", "Dólar", "PBR", "PBRA"]: 
        if col in daily_table.columns:
            s = daily_table[col].astype(float)
            norm_s = (s / s.iloc[0]) * 100.0 if len(s) > 0 and s.iloc[0] != 0 else pd.Series(dtype=float)
            if not norm_s.empty and not norm_s.isnull().all():
                 normalized_prices[col] = norm_s
            
    df_plot = pd.DataFrame(normalized_prices).join(daily_table["Sentiment"])
    df_prices_only = df_plot.drop(columns="Sentiment", errors="ignore") 

    if not df_prices_only.empty:
        fig_prices = px.line(
            df_prices_only, 
            title="Evolução dos Preços (Normalizado para 100 no início do período)",
            labels={"value": "Valor (Índice)", "index": "Data", "variable": "Ativo"},
            height=500
        )
        fig_prices.update_layout(legend_title_text='Ativo', hovermode="x unified")
        st.plotly_chart(fig_prices, use_container_width=True)
    else:
        st.info("Dados insuficientes para gerar o gráfico de preços.")

    st.markdown("---")

    st.subheader("Sentimento Diário Médio")
    sent_data = df_plot["Sentiment"].dropna()
    
    # 🌟 CORREÇÃO DE PLOTAGEM: Preenche NaN com 0.0 para garantir que o gráfico seja desenhado
    sent_data_plot = df_plot["Sentiment"].fillna(0.0) 
    
    if not sent_data_plot.empty:
        fig_sent = px.bar(
            sent_data_plot, # Usa a série com NaN preenchidos
            title="Score de Sentimento Diário",
            labels={"value": "Score", "index": "Data"},
            height=250,
            color=sent_data_plot,
            color_continuous_scale=px.colors.diverging.RdYlGn,
            range_color=[-1, 1]
        )
        fig_sent.update_layout(showlegend=False, coloraxis_showscale=False, hovermode="x unified")
        st.plotly_chart(fig_sent, use_container_width=True)
    else:
        st.info("Dados insuficientes para gerar o gráfico de sentimento.")

    st.markdown("---")

    # Tabela de Valores Diários Consolidados
    st.subheader("Valores Diários Consolidados (Fechamento e Sentimento)")
    if not daily_table.empty:
        st.dataframe(
            daily_table.style.format({
                "IBOV": "{:,.2f}",
                "PETR3": "{:.2f}",
                "PETR4": "{:.2f}",
                "PBR": "{:.2f}",
                "PBRA": "{:.2f}",
                "Brent": "{:.2f}",
                "Dólar": "{:.3f}",
                "Sentiment": "{:.3f}"
            }), 
            use_container_width=True
        )
    else:
         st.info("Nenhum dado diário consolidado disponível para o período selecionado.")

    st.markdown("---") 
    
    # Variações Diárias
    st.subheader("Variações Diárias (%)")
    variations_df = build_variations_table(daily_table)

    if not variations_df.empty:
        st.markdown("#### Últimas Variações de Fechamento")
        st.dataframe(
            variations_df.tail(7).style.format("{:.3f}%").apply(
                lambda x: ['background-color: #d4edda' if v > 0.1 else ('background-color: #f8d7da' if v < -0.1 else '') for v in x], axis=1
            ), 
            use_container_width=True
        )
    else:
        st.info("Nenhuma variação diária disponível.")


with tab2:
    st.markdown("## 📰 Análise de Notícias Detalhada")
    st.write(f"Total de **{len(news_processed)}** notícias encontradas sobre **PETROBRAS** (tema: **{selected_theme}**).")

    if news_processed:
        news_df = pd.DataFrame(news_processed)
        
        # Preparação dos dados para a tabela
        news_df['published_date_f'] = pd.to_datetime(news_df['published_date'], errors='coerce', utc=True).dt.tz_convert('America/Sao_Paulo').dt.strftime('%d/%m/%Y %H:%M')
        news_df['sentiment_score_f'] = news_df['sentiment_score'].apply(lambda x: f"{x:.3f}")
        
        df_display = news_df[[
            'published_date_f', 
            'title', 
            'publisher', 
            'category', 
            'sentiment',
            'sentiment_score_f',
            'url'
        ]].rename(columns={
            'published_date_f': 'Data/Hora',
            'title': 'Título da Notícia',
            'publisher': 'Fonte',
            'category': 'Categoria',
            'sentiment': 'Sentimento',
            'sentiment_score_f': 'Score',
            'url': 'Link'
        })

        st.data_editor(
            df_display,
            column_config={
                "Sentimento": st.column_config.Column("Sentimento", width="small"),
                "Título da Notícia": st.column_config.Column("Título da Notícia", width="large"),
                "Link": st.column_config.LinkColumn("Link", display_text="Abrir 🔗", width="small")
            },
            hide_index=True,
            use_container_width=True
        )
        
        @st.cache_data
        def convert_df_to_csv(df):
            cols_to_save = ['published_date', 'title', 'description', 'body', 'publisher', 'url', 'category', 'sentiment', 'sentiment_score']
            return df[cols_to_save].to_csv(index=False).encode('utf-8')

        csv = convert_df_to_csv(news_df)
        st.download_button(
            label="Baixar Dados de Notícias (CSV)",
            data=csv,
            file_name='petrobras_news_sentiment.csv',
            mime='text/csv',
        )
    else:
        st.info("Nenhuma notícia encontrada com os filtros selecionados.")

with tab3:
    st.markdown("## 🔗 Matriz de Correlação das Variações Diárias")
    st.write("Calcula a correlação entre as **variações diárias** dos ativos de mercado e o **Sentimento Diário Médio** das notícias.")
    
    # 1. Obter Variações de Mercado
    variations_market = calc_variations(stock_data, brent_df, dollar_df)
        
    # 2. Construir o dicionário para a correlação
    correlation_input = {}
    
    for col in variations_market.columns:
        correlation_input[col] = variations_market[col].fillna(0.0) 
        
    # Adicionar o Sentimento Diário (score)
    if "Sentiment" in daily_table.columns and not daily_table["Sentiment"].empty:
        
        sentiment_aligned = daily_table["Sentiment"].reindex(variations_market.index)
        correlation_input["Sentimento"] = sentiment_aligned.fillna(0.0)

    correlation_df = corr_table(correlation_input)
    
    if not correlation_df.empty:
        fig_corr = px.imshow(
            correlation_df,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale=px.colors.diverging.RdBu,
            zmin=-1, zmax=1,
            title="Correlação (Variações diárias vs. Score de Sentimento)",
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        st.markdown(f"> **Interpretação:** Valores próximos de **+1** (vermelho) indicam que ativos e sentimento se movem na mesma direção. Valores próximos de **-1** (azul) indicam que se movem em direções opostas.")
    else:
        st.info("Dados insuficientes para calcular a correlação.")
