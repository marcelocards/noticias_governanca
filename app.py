import os
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import numpy as np

# ──────────────────────────────────────────────
# Configurações
# ──────────────────────────────────────────────
TITLE_WEIGHT     = 0.7
BODY_WEIGHT      = 0.3
USE_ARTICLE_BODY = False

from core import (
    get_stock_data, get_brent_prices, get_dollar_rate,
    get_news, process_news_with_sentiment,
    daily_sentiment_series, build_daily_table, build_variations_table,
    CATEGORIES, calc_variations, corr_table,
)

# ──────────────────────────────────────────────
# Página
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Petrobras Radar",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* Tipografia geral */
[data-testid="stAppViewContainer"] { font-family: 'Inter', sans-serif; }

/* Cards de KPI */
.kpi-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.5rem;
    text-align: center;
}
.kpi-label  { font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
.kpi-value  { font-size: 26px; font-weight: 700; color: #f1f5f9; line-height: 1.1; }
.kpi-delta  { font-size: 13px; margin-top: 4px; }
.kpi-up     { color: #34d399; }
.kpi-down   { color: #f87171; }
.kpi-flat   { color: #94a3b8; }

/* Badges de sentimento */
.badge-pos { background:#064e3b; color:#6ee7b7; padding:2px 10px; border-radius:20px; font-size:12px; font-weight:600; }
.badge-neg { background:#7f1d1d; color:#fca5a5; padding:2px 10px; border-radius:20px; font-size:12px; font-weight:600; }
.badge-neu { background:#78350f; color:#fcd34d; padding:2px 10px; border-radius:20px; font-size:12px; font-weight:600; }

/* Card de notícia */
.news-card {
    border: 1px solid #1e293b;
    border-left: 4px solid #334155;
    border-radius: 8px;
    padding: 0.9rem 1rem;
    margin-bottom: 10px;
    background: #0f172a;
}
.news-card.pos { border-left-color: #34d399; }
.news-card.neg { border-left-color: #f87171; }
.news-card.neu { border-left-color: #fbbf24; }
.news-title { font-size: 14px; font-weight: 600; color: #e2e8f0; margin-bottom: 6px; line-height: 1.4; }
.news-meta  { font-size: 11px; color: #64748b; }
.news-score { float: right; font-size: 13px; font-weight: 700; }

/* Seção de interpretação */
.insight-box {
    background: #0f172a;
    border: 1px solid #1d4ed8;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
}
.insight-title { font-size: 13px; font-weight: 700; color: #93c5fd; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px; }
.insight-text  { font-size: 13px; color: #cbd5e1; line-height: 1.7; }

/* Divider */
.section-title { font-size: 16px; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 1.5px; margin: 1.5rem 0 0.75rem; }

/* Gauge */
.gauge-wrap { text-align: center; padding: 0.5rem; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# CACHE — Camada 1: dados brutos (1h)
# ──────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="🔄 Buscando dados de mercado...")
def load_raw_data(end_date, days_ago, query_theme):
    end_dt   = datetime.combine(end_date, datetime.min.time())
    start_dt = end_dt - timedelta(days=days_ago)
    tickers  = ["PETR3.SA", "PETR4.SA", "^BVSP", "PBR", "PBRA"]
    stock_data = get_stock_data(tickers, start_dt, end_dt)
    brent_df   = get_brent_prices(start_dt, end_dt)
    dollar_df  = get_dollar_rate(start_dt, end_dt)
    news_raw   = get_news("PETROBRAS", period=f"{days_ago}d", max_results=30, theme=query_theme)
    return stock_data, brent_df, dollar_df, news_raw


# ──────────────────────────────────────────────
# CACHE — Camada 2: sentimento (30min)
# ──────────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner="🧠 Analisando sentimento das notícias...")
def load_sentiment(news_raw_hashable, use_body, w_title, w_body):
    news_list      = [dict(item) for item in news_raw_hashable]
    news_processed = process_news_with_sentiment(news_list, use_body=use_body, w_title=w_title, w_body=w_body)
    daily_sent     = daily_sentiment_series(news_processed)
    return news_processed, daily_sent


# ──────────────────────────────────────────────
# Helpers de interpretação
# ──────────────────────────────────────────────
def interpret_sentiment(score: float) -> tuple[str, str, str]:
    """Retorna (label, classe_css, emoji)"""
    if score > 0.3:   return "Muito Positivo", "pos", "🟢"
    if score > 0.05:  return "Levemente Positivo", "pos", "🟡"
    if score < -0.3:  return "Muito Negativo", "neg", "🔴"
    if score < -0.05: return "Levemente Negativo", "neg", "🟠"
    return "Neutro", "neu", "⚪"


def sentiment_insight(score: float, n_news: int, n_neg: int, n_pos: int) -> str:
    pct_neg = (n_neg / n_news * 100) if n_news > 0 else 0
    pct_pos = (n_pos / n_news * 100) if n_news > 0 else 0
    if score > 0.3:
        return (f"O fluxo de notícias é predominantemente favorável à Petrobras neste período "
                f"({pct_pos:.0f}% das {n_news} notícias com tom positivo). Isso tende a reduzir "
                f"prêmio de risco e pode sustentar o preço das ações no curto prazo.")
    if score > 0.05:
        return (f"Notícias levemente positivas ({pct_pos:.0f}% positivas de {n_news} no período). "
                f"Ambiente favorável, mas sem catalisador forte. Acompanhe Brent e câmbio como drivers principais.")
    if score < -0.3:
        return (f"Clima de notícias muito negativo — {pct_neg:.0f}% das {n_news} notícias com tom "
                f"adverso. Situações assim historicamente aumentam volatilidade e pressionam PETR4. "
                f"Atenção especial à governança e decisões de pricing.")
    if score < -0.05:
        return (f"Notícias com viés negativo ({pct_neg:.0f}% de {n_news}). Ruídos moderados. "
                f"Verifique se as notícias negativas são pontuais ou refletem tendência de política corporativa.")
    return (f"Ambiente noticioso equilibrado ({n_news} notícias, {pct_pos:.0f}% positivas / {pct_neg:.0f}% negativas). "
            f"Preço das ações deve responder mais ao Brent e câmbio do que ao fluxo de notícias neste momento.")


def brent_dolar_insight(brent_var: float, dolar_var: float) -> str:
    parts = []
    if abs(brent_var) > 0.01:
        direcao = "subiu" if brent_var > 0 else "caiu"
        impacto = "positivo" if brent_var > 0 else "negativo"
        parts.append(f"Brent {direcao} {abs(brent_var):.1f}% no período — impacto {impacto} para receita em USD.")
    if abs(dolar_var) > 0.01:
        direcao = "se valorizou" if dolar_var > 0 else "se desvalorizou"
        efeito  = "amplifica receitas em R$" if dolar_var > 0 else "comprime margens em R$"
        parts.append(f"Dólar {direcao} {abs(dolar_var):.1f}% — câmbio mais alto {efeito}.")
    if brent_var > 0 and dolar_var > 0:
        parts.append("🔵 Combinação duplamente favorável: Brent em alta + dólar forte tende a impulsionar PETR4.")
    elif brent_var < 0 and dolar_var < 0:
        parts.append("🔴 Dupla pressão negativa: Brent em queda + dólar fraco comprime resultado em reais.")
    return " ".join(parts) if parts else "Variações de Brent e câmbio dentro da normalidade no período analisado."


def corr_insight(corr_df: pd.DataFrame) -> str:
    if corr_df.empty or "Sentimento" not in corr_df.columns:
        return ""
    for col in ["PETR4", "PETR3"]:
        if col in corr_df.index:
            val = corr_df.loc[col, "Sentimento"]
            if abs(val) > 0.3:
                direcao = "positiva" if val > 0 else "negativa"
                return (f"Correlação {direcao} ({val:.2f}) entre sentimento das notícias e variação de {col} "
                        f"no período — as notícias têm {'acompanhado' if val > 0 else 'precedido inversamente'} "
                        f"o movimento das ações.")
    return "Correlação baixa entre sentimento e ações neste período — Brent e câmbio explicam melhor os movimentos."


def make_gauge(score: float) -> go.Figure:
    color = "#34d399" if score > 0.05 else ("#f87171" if score < -0.05 else "#fbbf24")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score, 3),
        number={"font": {"size": 28, "color": "#f1f5f9"}, "suffix": ""},
        gauge={
            "axis": {"range": [-1, 1], "tickcolor": "#475569", "tickwidth": 1, "ticklen": 5,
                     "tickfont": {"size": 10, "color": "#94a3b8"}},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": "#1e293b",
            "borderwidth": 0,
            "steps": [
                {"range": [-1, -0.3], "color": "#450a0a"},
                {"range": [-0.3, -0.05], "color": "#7f1d1d"},
                {"range": [-0.05, 0.05], "color": "#1c1917"},
                {"range": [0.05, 0.3],  "color": "#064e3b"},
                {"range": [0.3, 1],     "color": "#052e16"},
            ],
            "threshold": {"line": {"color": "#f1f5f9", "width": 2}, "thickness": 0.8, "value": score},
        },
    ))
    fig.update_layout(
        height=180, margin=dict(t=10, b=5, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)", font_color="#f1f5f9",
    )
    return fig


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Petrobras_logo.svg/320px-Petrobras_logo.svg.png", width=140)
    st.markdown("### 🛢️ Petrobras Radar")
    st.markdown("---")

    st.markdown("**📅 Período**")
    today    = datetime.now().date()
    days_ago = st.slider("Dias de análise", min_value=7, max_value=180, value=60, step=7)
    end_date = st.date_input("Data final", value=today, max_value=today)

    st.markdown("**🗂️ Tema das notícias**")
    theme_options  = ["Todos"] + list(CATEGORIES.keys())
    selected_theme = st.selectbox("Filtrar por tema", options=theme_options, index=0, label_visibility="collapsed")

    st.markdown("**⚖️ Peso do sentimento**")
    title_weight_input = st.slider("Título vs Corpo", 0.0, 1.0, TITLE_WEIGHT, 0.1,
                                   help="1.0 = apenas título, 0.0 = apenas corpo do artigo")
    body_weight_input  = round(1.0 - title_weight_input, 1)
    st.caption(f"Título: **{title_weight_input:.1f}** | Corpo: **{body_weight_input:.1f}**")

    use_body_input = st.checkbox("Extrair corpo do artigo", value=False,
                                 help="Mais preciso, porém mais lento")

    st.markdown("---")
    st.caption("Dados: Yahoo Finance · BCB PTAX · Google News")


# ──────────────────────────────────────────────
# CARREGAMENTO
# ──────────────────────────────────────────────
stock_data, brent_df, dollar_df, news_raw = load_raw_data(end_date, days_ago, selected_theme)

news_raw_hashable = tuple(tuple(sorted(n.items())) for n in news_raw)
news_processed, daily_sent = load_sentiment(news_raw_hashable, use_body_input, title_weight_input, body_weight_input)

daily_table = build_daily_table(stock_data, brent_df, dollar_df, daily_sent)

if daily_table.empty and not news_processed:
    st.error("⚠️ Não foi possível carregar dados. Verifique sua conexão ou reduza o período.")
    st.stop()


# ──────────────────────────────────────────────
# PRÉ-CÁLCULOS GLOBAIS
# ──────────────────────────────────────────────
news_df = pd.DataFrame(news_processed) if news_processed else pd.DataFrame()
avg_sentiment  = float(news_df["sentiment_score"].mean()) if not news_df.empty and "sentiment_score" in news_df.columns else 0.0
n_pos = int((news_df["sentiment"] == "Positivo").sum()) if not news_df.empty else 0
n_neg = int((news_df["sentiment"] == "Negativo").sum()) if not news_df.empty else 0
n_neu = int((news_df["sentiment"] == "Neutro").sum())   if not news_df.empty else 0
n_tot = len(news_df)

sent_label, sent_css, sent_emoji = interpret_sentiment(avg_sentiment)

# Variações do período
def _period_var(series: pd.Series) -> float:
    s = series.dropna()
    if len(s) < 2 or s.iloc[0] == 0:
        return 0.0
    return float((s.iloc[-1] - s.iloc[0]) / s.iloc[0] * 100)

petr4_var  = _period_var(daily_table["PETR4"])  if "PETR4"  in daily_table.columns else 0.0
petr3_var  = _period_var(daily_table["PETR3"])  if "PETR3"  in daily_table.columns else 0.0
brent_var  = _period_var(daily_table["Brent"])  if "Brent"  in daily_table.columns else 0.0
dolar_var  = _period_var(daily_table["Dólar"])  if "Dólar"  in daily_table.columns else 0.0
ibov_var   = _period_var(daily_table["IBOV"])   if "IBOV"   in daily_table.columns else 0.0

def _last(col):
    if col not in daily_table.columns: return None
    s = daily_table[col].dropna()
    return float(s.iloc[-1]) if len(s) > 0 else None

petr4_last = _last("PETR4")
petr3_last = _last("PETR3")
brent_last = _last("Brent")
dolar_last = _last("Dólar")


# ──────────────────────────────────────────────
# CABEÇALHO
# ──────────────────────────────────────────────
st.markdown(f"## 🛢️ Petrobras Radar &nbsp;&nbsp;<span style='font-size:14px;color:#64748b;font-weight:400'>Últimos {days_ago} dias até {end_date.strftime('%d/%m/%Y')}</span>", unsafe_allow_html=True)
st.markdown("---")


# ──────────────────────────────────────────────
# LINHA DE KPIs
# ──────────────────────────────────────────────
def kpi_html(label, value, delta, prefix="", suffix=""):
    if value is None:
        return f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value" style="font-size:18px;color:#475569">—</div></div>'
    d_class = "kpi-up" if delta > 0 else ("kpi-down" if delta < 0 else "kpi-flat")
    d_arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "—")
    return f'''<div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{prefix}{value:,.2f}{suffix}</div>
        <div class="kpi-delta {d_class}">{d_arrow} {abs(delta):.1f}% no período</div>
    </div>'''

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1: st.markdown(kpi_html("PETR4", petr4_last, petr4_var, "R$ "), unsafe_allow_html=True)
with c2: st.markdown(kpi_html("PETR3", petr3_last, petr3_var, "R$ "), unsafe_allow_html=True)
with c3: st.markdown(kpi_html("Brent (US$)", brent_last, brent_var, "$ "), unsafe_allow_html=True)
with c4: st.markdown(kpi_html("Dólar (R$)", dolar_last, dolar_var, "R$ "), unsafe_allow_html=True)
with c5:
    sent_color = "#34d399" if avg_sentiment > 0.05 else ("#f87171" if avg_sentiment < -0.05 else "#fbbf24")
    st.markdown(f'''<div class="kpi-card">
        <div class="kpi-label">Sentimento Notícias</div>
        <div class="kpi-value" style="font-size:20px;color:{sent_color}">{sent_emoji} {sent_label}</div>
        <div class="kpi-delta kpi-flat">{n_tot} notícias analisadas</div>
    </div>''', unsafe_allow_html=True)
with c6:
    ibov_color = "#34d399" if ibov_var > 0 else "#f87171"
    ibov_last_v = _last("IBOV")
    st.markdown(kpi_html("IBOV", ibov_last_v, ibov_var), unsafe_allow_html=True)

st.markdown("")


# ──────────────────────────────────────────────
# ABAS
# ──────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Mercado & Preços", "📰 Notícias & Sentimento", "🔗 Correlação & Insights"])


# ════════════════════════════════════════════════
# TAB 1 — MERCADO
# ════════════════════════════════════════════════
with tab1:

    # ── Gráfico principal: preços normalizados + sentimento sobreposto ──
    st.markdown('<div class="section-title">Evolução de Preços × Sentimento</div>', unsafe_allow_html=True)

    normalized = {}
    for col in ["PETR4", "PETR3", "Brent", "Dólar", "IBOV"]:
        if col in daily_table.columns:
            s = daily_table[col].astype(float).dropna()
            if len(s) > 1 and s.iloc[0] != 0:
                normalized[col] = (s / s.iloc[0]) * 100.0

    if normalized:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.70, 0.30],
                            vertical_spacing=0.04,
                            subplot_titles=("Preços normalizados (base 100)", "Sentimento diário"))

        colors = {"PETR4": "#3b82f6", "PETR3": "#60a5fa", "Brent": "#f59e0b",
                  "Dólar": "#a78bfa", "IBOV": "#94a3b8"}

        for name, series in normalized.items():
            fig.add_trace(go.Scatter(
                x=series.index, y=series.values, name=name,
                line=dict(color=colors.get(name, "#e2e8f0"), width=2),
                hovertemplate=f"<b>{name}</b>: %{{y:.1f}}<extra></extra>",
            ), row=1, col=1)

        # Sentimento como barras coloridas
        if "Sentiment" in daily_table.columns:
            sent_s = daily_table["Sentiment"].dropna()
            if not sent_s.empty:
                bar_colors = ["#34d399" if v > 0.05 else ("#f87171" if v < -0.05 else "#fbbf24") for v in sent_s]
                fig.add_trace(go.Bar(
                    x=sent_s.index, y=sent_s.values, name="Sentimento",
                    marker_color=bar_colors, opacity=0.85,
                    hovertemplate="<b>Sentimento</b>: %{y:.3f}<extra></extra>",
                ), row=2, col=1)
                fig.add_hline(y=0, line_dash="dot", line_color="#475569", row=2, col=1)

        fig.update_layout(
            height=520, hovermode="x unified",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0f172a",
            font=dict(color="#94a3b8", size=12),
            legend=dict(orientation="h", y=1.06, x=0, bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=40, b=20, l=10, r=10),
        )
        for ax in ["xaxis", "xaxis2", "yaxis", "yaxis2"]:
            fig.update_layout(**{ax: dict(gridcolor="#1e293b", zerolinecolor="#334155")})

        st.plotly_chart(fig, use_container_width=True)

    # ── Insight Brent + Dólar ──
    insight_bd = brent_dolar_insight(brent_var, dolar_var)
    st.markdown(f'''<div class="insight-box">
        <div class="insight-title">📌 Brent & Câmbio — O que isso significa?</div>
        <div class="insight-text">{insight_bd}</div>
    </div>''', unsafe_allow_html=True)

    # ── Comparativo de retorno no período ──
    st.markdown('<div class="section-title">Retorno no Período (%)</div>', unsafe_allow_html=True)

    retornos = {k: v for k, v in [
        ("PETR4", petr4_var), ("PETR3", petr3_var),
        ("Brent", brent_var), ("Dólar", dolar_var), ("IBOV", ibov_var),
    ] if v != 0.0}

    if retornos:
        df_ret = pd.DataFrame({"Ativo": list(retornos.keys()), "Retorno (%)": list(retornos.values())})
        df_ret = df_ret.sort_values("Retorno (%)", ascending=True)
        bar_clrs = ["#34d399" if v > 0 else "#f87171" for v in df_ret["Retorno (%)"]]
        fig_ret = px.bar(df_ret, x="Retorno (%)", y="Ativo", orientation="h",
                         text=df_ret["Retorno (%)"].apply(lambda x: f"{x:+.1f}%"),
                         height=220)
        fig_ret.update_traces(marker_color=bar_clrs, textposition="outside")
        fig_ret.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0f172a",
            font=dict(color="#94a3b8"), margin=dict(t=10, b=10, l=10, r=60),
            xaxis=dict(gridcolor="#1e293b", zerolinecolor="#475569"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_ret, use_container_width=True)

    # ── Tabela de variações diárias (últimos 10 dias) ──
    st.markdown('<div class="section-title">Variações Diárias — Últimos 10 Dias (%)</div>', unsafe_allow_html=True)
    var_df = build_variations_table(daily_table)
    if not var_df.empty:
        show_cols = [c for c in ["PETR4", "PETR3", "Brent", "Dólar", "IBOV"] if c in var_df.columns]
        var_show  = var_df[show_cols].tail(10).copy()

        def color_cell(val):
            if pd.isna(val): return ""
            if val > 0.5:    return "background-color:#052e16;color:#6ee7b7"
            if val > 0:      return "background-color:#064e3b33;color:#a7f3d0"
            if val < -0.5:   return "background-color:#450a0a;color:#fca5a5"
            if val < 0:      return "background-color:#7f1d1d33;color:#fecaca"
            return "color:#94a3b8"

        styled = var_show.style.format("{:+.2f}%").applymap(color_cell)
        st.dataframe(styled, use_container_width=True)


# ════════════════════════════════════════════════
# TAB 2 — NOTÍCIAS & SENTIMENTO
# ════════════════════════════════════════════════
with tab2:

    if news_df.empty:
        st.info("Nenhuma notícia encontrada. Tente alterar o tema ou ampliar o período.")
    else:
        # ── Painel de sentimento geral ──
        col_gauge, col_dist = st.columns([1, 2])

        with col_gauge:
            st.markdown('<div class="section-title">Termômetro de Sentimento</div>', unsafe_allow_html=True)
            st.plotly_chart(make_gauge(avg_sentiment), use_container_width=True)
            st.markdown(f'<div style="text-align:center;font-size:15px;font-weight:700;color:#e2e8f0">{sent_emoji} {sent_label}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="text-align:center;font-size:12px;color:#64748b;margin-top:4px">Score médio: {avg_sentiment:+.3f}</div>', unsafe_allow_html=True)

        with col_dist:
            st.markdown('<div class="section-title">Distribuição por Sentimento</div>', unsafe_allow_html=True)
            fig_dist = go.Figure()
            cat_data = {"Positivo": n_pos, "Neutro": n_neu, "Negativo": n_neg}
            clrs     = {"Positivo": "#34d399", "Neutro": "#fbbf24", "Negativo": "#f87171"}
            for label, count in cat_data.items():
                if count > 0:
                    fig_dist.add_trace(go.Bar(
                        x=[label], y=[count], name=label,
                        marker_color=clrs[label],
                        text=[f"{count} notícias\n{count/n_tot*100:.0f}%"],
                        textposition="inside",
                        hovertemplate=f"<b>{label}</b>: {count}<extra></extra>",
                    ))
            fig_dist.update_layout(
                height=210, showlegend=False, barmode="group",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0f172a",
                font=dict(color="#94a3b8"), margin=dict(t=10, b=10, l=10, r=10),
                xaxis=dict(gridcolor="rgba(0,0,0,0)"),
                yaxis=dict(gridcolor="#1e293b"),
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        # ── Insight de sentimento ──
        insight_sent = sentiment_insight(avg_sentiment, n_tot, n_neg, n_pos)
        st.markdown(f'''<div class="insight-box">
            <div class="insight-title">🧠 O que o sentimento indica?</div>
            <div class="insight-text">{insight_sent}</div>
        </div>''', unsafe_allow_html=True)

        st.markdown("---")

        # ── Filtros de notícias ──
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            sent_filter = st.multiselect("Sentimento", ["Positivo", "Neutro", "Negativo"],
                                         default=["Positivo", "Neutro", "Negativo"])
        with col_f2:
            cats = sorted(news_df["category"].unique().tolist()) if "category" in news_df.columns else []
            cat_filter = st.multiselect("Categoria", cats, default=cats)
        with col_f3:
            sort_opt = st.selectbox("Ordenar por", ["Mais recentes", "Mais negativos", "Mais positivos"])

        filtered = news_df.copy()
        if sent_filter and "sentiment" in filtered.columns:
            filtered = filtered[filtered["sentiment"].isin(sent_filter)]
        if cat_filter and "category" in filtered.columns:
            filtered = filtered[filtered["category"].isin(cat_filter)]

        filtered["published_date_dt"] = pd.to_datetime(filtered["published_date"], errors="coerce", utc=True)
        if sort_opt == "Mais recentes":
            filtered = filtered.sort_values("published_date_dt", ascending=False)
        elif sort_opt == "Mais negativos":
            filtered = filtered.sort_values("sentiment_score", ascending=True)
        else:
            filtered = filtered.sort_values("sentiment_score", ascending=False)

        st.markdown(f'<div class="section-title">Notícias ({len(filtered)} de {n_tot})</div>', unsafe_allow_html=True)

        # ── Cards de notícias ──
        for _, row in filtered.head(25).iterrows():
            score = row.get("sentiment_score", 0)
            css   = "pos" if score > 0.05 else ("neg" if score < -0.05 else "neu")
            dt_raw = row.get("published_date_dt", None)
            dt_str = ""
            try:
                if pd.notna(dt_raw):
                    dt_str = pd.Timestamp(dt_raw).tz_convert("America/Sao_Paulo").strftime("%d/%m/%Y %H:%M")
            except Exception:
                pass

            badge_html = (f'<span class="badge-pos">Positivo</span>' if css == "pos"
                          else (f'<span class="badge-neg">Negativo</span>' if css == "neg"
                                else f'<span class="badge-neu">Neutro</span>'))
            score_color = "#34d399" if score > 0.05 else ("#f87171" if score < -0.05 else "#fbbf24")

            title  = row.get("title", "Sem título")
            url    = row.get("url", "#")
            pub    = row.get("publisher", "")
            cat    = row.get("category", "")

            st.markdown(f'''<div class="news-card {css}">
                <span class="news-score" style="color:{score_color}">{score:+.2f}</span>
                <div class="news-title"><a href="{url}" target="_blank" style="color:#e2e8f0;text-decoration:none">{title}</a></div>
                <div class="news-meta">{badge_html} &nbsp; {pub} &nbsp;·&nbsp; {cat} &nbsp;·&nbsp; {dt_str}</div>
            </div>''', unsafe_allow_html=True)

        # ── Distribuição por categoria ──
        if "category" in news_df.columns:
            st.markdown('<div class="section-title">Distribuição por Tema</div>', unsafe_allow_html=True)
            cat_counts = news_df.groupby(["category", "sentiment"]).size().reset_index(name="count")
            fig_cat = px.bar(cat_counts, x="category", y="count", color="sentiment",
                             color_discrete_map={"Positivo": "#34d399", "Negativo": "#f87171", "Neutro": "#fbbf24"},
                             height=280, barmode="stack",
                             labels={"category": "", "count": "Notícias", "sentiment": ""})
            fig_cat.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0f172a",
                font=dict(color="#94a3b8"), margin=dict(t=10, b=10, l=10, r=10),
                xaxis=dict(gridcolor="rgba(0,0,0,0)", tickangle=-20),
                yaxis=dict(gridcolor="#1e293b"),
                legend=dict(orientation="h", y=1.1),
            )
            st.plotly_chart(fig_cat, use_container_width=True)

        # ── Export ──
        @st.cache_data
        def to_csv(df):
            cols = [c for c in ["published_date","title","publisher","category","sentiment","sentiment_score","url"] if c in df.columns]
            return df[cols].to_csv(index=False).encode("utf-8")

        st.download_button("⬇️ Exportar notícias (CSV)", to_csv(news_df),
                           "petrobras_noticias.csv", "text/csv")


# ════════════════════════════════════════════════
# TAB 3 — CORRELAÇÃO & INSIGHTS
# ════════════════════════════════════════════════
with tab3:

    st.markdown('<div class="section-title">Correlação: Sentimento × Mercado</div>', unsafe_allow_html=True)

    variations_market = calc_variations(stock_data, brent_df, dollar_df)
    correlation_input = {col: variations_market[col] for col in variations_market.columns}

    if "Sentiment" in daily_table.columns and not daily_table["Sentiment"].dropna().empty:
        correlation_input["Sentimento"] = daily_table["Sentiment"].reindex(variations_market.index)

    correlation_df, n_valid = corr_table(correlation_input)

    if not correlation_df.empty:
        # ── Insight de correlação ──
        c_ins = corr_insight(correlation_df)
        if c_ins:
            st.markdown(f'''<div class="insight-box">
                <div class="insight-title">🔗 O que a correlação revela?</div>
                <div class="insight-text">{c_ins}</div>
            </div>''', unsafe_allow_html=True)

        st.caption(f"Calculado com **{n_valid} dias** com dados completos para todos os ativos.")
        if n_valid < 10:
            st.warning(f"⚠️ Apenas {n_valid} dias — aumente o período para resultados mais robustos.")

        fig_corr = px.imshow(
            correlation_df.round(2),
            text_auto=True,
            aspect="auto",
            color_continuous_scale=[[0, "#1d4ed8"], [0.5, "#0f172a"], [1, "#15803d"]],
            zmin=-1, zmax=1,
            height=420,
        )
        fig_corr.update_traces(textfont_size=13)
        fig_corr.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8"),
            margin=dict(t=20, b=20, l=20, r=20),
            coloraxis_colorbar=dict(tickfont=dict(color="#94a3b8"), title="Corr."),
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        # ── Tabela de correlações com Sentimento ordenada ──
        if "Sentimento" in correlation_df.columns:
            st.markdown('<div class="section-title">Força da Relação com o Sentimento</div>', unsafe_allow_html=True)
            corr_sent = correlation_df["Sentimento"].drop("Sentimento", errors="ignore").sort_values(key=abs, ascending=False)
            df_cs = corr_sent.reset_index()
            df_cs.columns = ["Ativo", "Correlação com Sentimento"]

            def interpret_corr(val):
                a = abs(val)
                if a > 0.6:   return "Forte"
                if a > 0.35:  return "Moderada"
                if a > 0.15:  return "Fraca"
                return "Desprezível"

            df_cs["Intensidade"]  = df_cs["Correlação com Sentimento"].apply(interpret_corr)
            df_cs["Direção"]      = df_cs["Correlação com Sentimento"].apply(lambda v: "↑ Mesma direção" if v > 0 else "↓ Direção oposta")

            def color_corr(val):
                if val > 0.35:  return "color:#34d399;font-weight:700"
                if val > 0:     return "color:#6ee7b7"
                if val < -0.35: return "color:#f87171;font-weight:700"
                return "color:#fca5a5"

            st.dataframe(
                df_cs.style
                    .format({"Correlação com Sentimento": "{:+.3f}"})
                    .applymap(color_corr, subset=["Correlação com Sentimento"]),
                use_container_width=True, hide_index=True,
            )

        # ── Sentimento diário vs PETR4 ──
        st.markdown('<div class="section-title">Sentimento vs. PETR4 ao Longo do Tempo</div>', unsafe_allow_html=True)
        if "Sentiment" in daily_table.columns and "PETR4" in daily_table.columns:
            df_ov = daily_table[["PETR4", "Sentiment"]].dropna()
            if not df_ov.empty:
                fig_ov = make_subplots(specs=[[{"secondary_y": True}]])
                fig_ov.add_trace(go.Scatter(
                    x=df_ov.index, y=df_ov["PETR4"],
                    name="PETR4 (R$)", line=dict(color="#3b82f6", width=2),
                    hovertemplate="<b>PETR4</b>: R$ %{y:.2f}<extra></extra>",
                ), secondary_y=False)
                bar_c = ["#34d399" if v > 0.05 else ("#f87171" if v < -0.05 else "#fbbf24") for v in df_ov["Sentiment"]]
                fig_ov.add_trace(go.Bar(
                    x=df_ov.index, y=df_ov["Sentiment"],
                    name="Sentimento", marker_color=bar_c, opacity=0.6,
                    hovertemplate="<b>Sentimento</b>: %{y:.3f}<extra></extra>",
                ), secondary_y=True)
                fig_ov.update_layout(
                    height=340, hovermode="x unified",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0f172a",
                    font=dict(color="#94a3b8"), margin=dict(t=10, b=10, l=10, r=10),
                    legend=dict(orientation="h", y=1.1, bgcolor="rgba(0,0,0,0)"),
                    xaxis=dict(gridcolor="#1e293b"),
                    yaxis=dict(gridcolor="#1e293b", title="PETR4 (R$)"),
                    yaxis2=dict(title="Sentimento", gridcolor="rgba(0,0,0,0)"),
                )
                st.plotly_chart(fig_ov, use_container_width=True)
    else:
        st.info("Dados insuficientes para calcular a correlação. Amplie o período de análise.")



