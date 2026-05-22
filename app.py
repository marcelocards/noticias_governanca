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
BODY_WEIGHT  = 0.3
USE_ARTICLE_BODY = False
MAX_BODY_CHARS   = 4000

# ==============================
# Imports do core
# ==============================
from core import (
    get_stock_data, get_brent_prices, get_dollar_rate,
    get_news, process_news_with_sentiment,
    daily_sentiment_series, build_daily_table, build_variations_table,
    CATEGORIES, calc_variations, corr_table,
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
.neutro   { color:#92400e; background:#fff7ed; }
</style>
""", unsafe_allow_html=True)


# ==============================
# CAMADA 1 — Dados Brutos (TTL 1h)
# Não depende de pesos de sentimento.
# ==============================
@st.cache_data(ttl=3600, show_spinner="Buscando dados de mercado e notícias brutas...")
def load_raw_data(
    end_date:   datetime,
    days_ago:   int,
    query_theme: str,
) -> tuple:
    """
    Busca dados de mercado e notícias brutas (sem sentimento).
    Cache de 1 hora — mudanças nos pesos NÃO invalidam este cache.
    """
    end_dt   = datetime.combine(end_date, datetime.min.time())
    start_dt = end_dt - timedelta(days=days_ago)

    tickers    = ["PETR3.SA", "PETR4.SA", "^BVSP", "PBR", "PBRA"]
    stock_data = get_stock_data(tickers, start_dt, end_dt)
    brent_df   = get_brent_prices(start_dt, end_dt)
    dollar_df  = get_dollar_rate(start_dt, end_dt)

    period_str = f"{days_ago}d"
    news_raw   = get_news(
        query="PETROBRAS",
        period=period_str,
        max_results=30,        # limite seguro para Streamlit Cloud
        theme=query_theme,
    )

    return stock_data, brent_df, dollar_df, news_raw


# ==============================
# CAMADA 2 — Sentimento (TTL 30min)
# Depende dos pesos — cache separado e mais curto.
# ==============================
@st.cache_data(ttl=1800, show_spinner="Calculando sentimento das notícias...")
def load_sentiment(
    news_raw:  list,
    use_body:  bool,
    w_title:   float,
    w_body:    float,
) -> tuple:
    """
    Processa sentimento e constrói as séries diárias.
    Cache de 30 min — invalida quando os pesos mudam.
    """
    # news_raw chega como tuple of tuples (para hashability do cache)
    # Precisa ser convertido de volta para lista de dicts antes de processar
    news_list = [dict(item) for item in news_raw]

    news_processed = process_news_with_sentiment(
        news_list,
        use_body=use_body,
        w_title=w_title,
        w_body=w_body,
    )
    daily_sent = daily_sentiment_series(news_processed)
    return news_processed, daily_sent


# ==============================
# Interface — Sidebar
# ==============================
st.title("📈 Petrobras – Mercado, Notícias e Análise de Sentimento")
st.sidebar.header("⚙️ Configurações da Análise")

today    = datetime.now().date()
days_ago = st.sidebar.slider("Período de Análise (Dias)", min_value=7, max_value=180, value=30, step=7)
end_date = st.sidebar.date_input("Data Final", value=today, max_value=today)

theme_options  = ["Todos"] + list(CATEGORIES.keys())
selected_theme = st.sidebar.selectbox("Filtrar Notícias por Tema", options=theme_options, index=0)

st.sidebar.subheader("Ajustes de Sentimento")
use_body_input = st.sidebar.checkbox(
    "Incluir corpo da notícia na análise (Lento)",
    value=USE_ARTICLE_BODY,
    help="Extrai o corpo do artigo via Newspaper3k. Aumenta o tempo de carregamento.",
)
title_weight_input = st.sidebar.slider(
    "Peso do Título (vs. Corpo)",
    min_value=0.0, max_value=1.0,
    value=TITLE_WEIGHT, step=0.1,
)
body_weight_input = 1.0 - title_weight_input
st.sidebar.markdown(f"*(Peso do Corpo: **{body_weight_input:.1f}**)*")

# ==============================
# Carregamento em duas etapas
# ==============================
stock_data, brent_df, dollar_df, news_raw = load_raw_data(
    end_date=end_date,
    days_ago=days_ago,
    query_theme=selected_theme,
)

# news_raw precisa ser hashável para o cache — usa tuple of frozensets internamente
news_raw_hashable = tuple(
    tuple(sorted(n.items())) for n in news_raw
)

news_processed, daily_sent = load_sentiment(
    news_raw=news_raw_hashable,
    use_body=use_body_input,
    w_title=title_weight_input,
    w_body=body_weight_input,
)

# Monta a tabela diária (não precisa de cache — é rápido)
daily_table = build_daily_table(stock_data, brent_df, dollar_df, daily_sent)

if daily_table.empty and not news_processed:
    st.error("Não foi possível carregar dados para o período selecionado. Tente reduzir o período.")
    st.stop()

# ==============================
# Abas
# ==============================
tab1, tab2, tab3 = st.tabs([
    "📊 Dashboard de Mercado e Sentimento",
    "📰 Análise de Notícias Detalhada",
    "🔗 Correlação",
])


# ─────────────────────────────
# TAB 1 — Dashboard
# ─────────────────────────────
with tab1:
    st.markdown("## 📊 Dashboard de Mercado e Sentimento")
    st.write(f"Análise dos últimos **{days_ago} dias** (até **{end_date.strftime('%d/%m/%Y')}**).")

    # KPIs no topo
    if not daily_table.empty:
        last_row  = daily_table.dropna(how="all").iloc[-1]
        first_row = daily_table.dropna(how="all").iloc[0]

        kpi_cols = st.columns(5)
        kpi_map  = [
            ("PETR4", "PETR4 (R$)"),
            ("PETR3", "PETR3 (R$)"),
            ("Brent", "Brent (US$)"),
            ("Dólar", "Dólar (R$)"),
            ("Sentiment", "Sentimento Médio"),
        ]
        for col, (key, label) in zip(kpi_cols, kpi_map):
            with col:
                val = last_row.get(key, None)
                if pd.notna(val):
                    prev = first_row.get(key, None)
                    delta_str = ""
                    if pd.notna(prev) and prev != 0:
                        pct = (val - prev) / prev * 100
                        delta_str = f"{pct:+.1f}%"
                    fmt = f"{val:,.3f}" if key in ("Dólar", "Sentiment") else f"{val:,.2f}"
                    st.metric(label, fmt, delta_str)

    st.markdown("---")

    # Gráfico de preços normalizados
    st.subheader("Preços Normalizados (Índice 100)")
    normalized_prices = {}
    for col in ["IBOV", "PETR4", "PETR3", "Brent", "Dólar", "PBR", "PBRA"]:
        if col in daily_table.columns:
            s      = daily_table[col].astype(float)
            norm_s = (s / s.iloc[0]) * 100.0 if len(s) > 0 and s.iloc[0] != 0 else pd.Series(dtype=float)
            if not norm_s.empty and not norm_s.isnull().all():
                normalized_prices[col] = norm_s

    df_plot       = pd.DataFrame(normalized_prices).join(daily_table.get("Sentiment", pd.Series(dtype=float)))
    df_prices_only = df_plot.drop(columns="Sentiment", errors="ignore")

    if not df_prices_only.empty:
        fig_prices = px.line(
            df_prices_only,
            title="Evolução dos Preços (Normalizado para 100 no início do período)",
            labels={"value": "Valor (Índice)", "index": "Data", "variable": "Ativo"},
            height=500,
        )
        fig_prices.update_layout(legend_title_text="Ativo", hovermode="x unified")
        st.plotly_chart(fig_prices, use_container_width=True)
    else:
        st.info("Dados insuficientes para gerar o gráfico de preços.")

    st.markdown("---")

    # Sentimento diário
    st.subheader("Sentimento Diário Médio")
    sent_data_plot = df_plot["Sentiment"].dropna() if "Sentiment" in df_plot.columns else pd.Series(dtype=float)

    if not sent_data_plot.empty:
        fig_sent = px.bar(
            sent_data_plot,
            title="Score de Sentimento Diário (apenas dias com notícias)",
            labels={"value": "Score", "index": "Data"},
            height=250,
            color=sent_data_plot,
            color_continuous_scale=px.colors.diverging.RdYlGn,
            range_color=[-1, 1],
        )
        fig_sent.update_xaxes(type="category")
        fig_sent.update_layout(showlegend=False, coloraxis_showscale=False, hovermode="x unified")
        st.plotly_chart(fig_sent, use_container_width=True)
    else:
        st.info("Nenhuma notícia com score encontrada no período selecionado.")

    st.markdown("---")

    # Tabela consolidada
    st.subheader("Valores Diários Consolidados")
    if not daily_table.empty:
        fmt = {
            "IBOV":      "{:,.2f}",
            "PETR3":     "{:.2f}",
            "PETR4":     "{:.2f}",
            "PBR":       "{:.2f}",
            "PBRA":      "{:.2f}",
            "Brent":     "{:.2f}",
            "Dólar":     "{:.3f}",
            "Sentiment": "{:.3f}",
        }
        existing_fmt = {k: v for k, v in fmt.items() if k in daily_table.columns}
        st.dataframe(daily_table.style.format(existing_fmt), use_container_width=True)

    st.markdown("---")

    # Variações diárias
    st.subheader("Variações Diárias (%)")
    variations_df = build_variations_table(daily_table)
    if not variations_df.empty:
        st.markdown("#### Últimas variações de fechamento")
        st.dataframe(
            variations_df.tail(7).style.format("{:.3f}%").apply(
                lambda x: [
                    "background-color: #d4edda" if v > 0.1
                    else ("background-color: #f8d7da" if v < -0.1 else "")
                    for v in x
                ],
                axis=1,
            ),
            use_container_width=True,
        )
    else:
        st.info("Nenhuma variação diária disponível.")


# ─────────────────────────────
# TAB 2 — Notícias Detalhadas
# ─────────────────────────────
with tab2:
    st.markdown("## 📰 Análise de Notícias Detalhada")
    st.write(f"Total de **{len(news_processed)}** notícias encontradas (tema: **{selected_theme}**).")

    if news_processed:
        news_df = pd.DataFrame(news_processed)
        news_df["published_date_dt"] = pd.to_datetime(news_df["published_date"], errors="coerce", utc=True)
        news_df = news_df.sort_values(by="published_date_dt", ascending=False)
        news_df["published_date_f"]  = (
            news_df["published_date_dt"]
            .dt.tz_convert("America/Sao_Paulo")
            .dt.strftime("%d/%m/%Y %H:%M")
        )
        news_df["sentiment_score_f"] = news_df["sentiment_score"].apply(lambda x: f"{x:.3f}")

        df_display = news_df[[
            "published_date_f", "title", "publisher",
            "category", "sentiment", "sentiment_score_f", "url",
        ]].rename(columns={
            "published_date_f":  "Data/Hora",
            "title":             "Título da Notícia",
            "publisher":         "Fonte",
            "category":          "Categoria",
            "sentiment":         "Sentimento",
            "sentiment_score_f": "Score",
            "url":               "Link",
        })

        st.data_editor(
            df_display,
            column_config={
                "Sentimento":        st.column_config.Column("Sentimento", width="small"),
                "Título da Notícia": st.column_config.Column("Título da Notícia", width="large"),
                "Link":              st.column_config.LinkColumn("Link", display_text="Abrir 🔗", width="small"),
            },
            hide_index=True,
            use_container_width=True,
        )

        # Distribuição de sentimentos
        sent_counts = news_df["sentiment"].value_counts()
        fig_pie = px.pie(
            values=sent_counts.values,
            names=sent_counts.index,
            title="Distribuição de Sentimento das Notícias",
            color=sent_counts.index,
            color_discrete_map={"Positivo": "#10b981", "Negativo": "#ef4444", "Neutro": "#f59e0b"},
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        @st.cache_data
        def convert_df_to_csv(df):
            cols = ["published_date", "title", "description", "body",
                    "publisher", "url", "category", "sentiment", "sentiment_score"]
            existing = [c for c in cols if c in df.columns]
            return df[existing].to_csv(index=False).encode("utf-8")

        csv = convert_df_to_csv(news_df)
        st.download_button(
            label="⬇️ Baixar Dados de Notícias (CSV)",
            data=csv,
            file_name="petrobras_news_sentiment.csv",
            mime="text/csv",
        )
    else:
        st.info("Nenhuma notícia encontrada com os filtros selecionados.")


# ─────────────────────────────
# TAB 3 — Correlação
# ─────────────────────────────
with tab3:
    st.markdown("## 🔗 Matriz de Correlação das Variações Diárias")
    st.write(
        "Correlação entre **variações diárias** dos ativos e o **score de sentimento** das notícias. "
        "Apenas dias com dados completos são usados (sem imputação de zeros)."
    )

    variations_market = calc_variations(stock_data, brent_df, dollar_df)
    correlation_input = {}

    for col in variations_market.columns:
        correlation_input[col] = variations_market[col]   # sem fillna!

    if "Sentiment" in daily_table.columns and not daily_table["Sentiment"].dropna().empty:
        sentiment_aligned = daily_table["Sentiment"].reindex(variations_market.index)
        correlation_input["Sentimento"] = sentiment_aligned  # NaN onde não há notícia

    correlation_df, n_valid = corr_table(correlation_input)

    if not correlation_df.empty:
        st.caption(f"ℹ️ Calculado com **{n_valid} dias** que têm dados completos para todas as variáveis.")

        fig_corr = px.imshow(
            correlation_df,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale=px.colors.diverging.RdBu,
            zmin=-1, zmax=1,
            title="Correlação (Variações diárias × Score de Sentimento)",
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown(
            "> **Interpretação:** Valores próximos de **+1** (vermelho) indicam que os ativos "
            "e o sentimento se movem na **mesma direção**. Valores próximos de **−1** (azul) "
            "indicam **direções opostas**. Valores próximos de 0 indicam baixa relação linear."
        )

        if n_valid < 10:
            st.warning(
                f"⚠️ Apenas {n_valid} dias com dados completos — a correlação pode não ser estatisticamente representativa. "
                "Aumente o período de análise para resultados mais confiáveis."
            )
    else:
        st.info("Dados insuficientes para calcular a correlação. Tente um período maior ou verifique as notícias encontradas.")


