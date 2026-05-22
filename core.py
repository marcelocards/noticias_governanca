import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from gnews import GNews

# =========================
# Config – pesos padrão
# =========================
TITLE_WEIGHT_DEFAULT = 0.7
BODY_WEIGHT_DEFAULT = 0.3

# =========================
# Mercado (Dados Reais)
# =========================

def get_stock_data(tickers: List[str], start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
    data = {}
    for ticker in tickers:
        try:
            tk = yf.Ticker(ticker)
            df = tk.history(start=start_date, end=end_date + timedelta(days=1))

            if df is None or df.empty:
                df = tk.history(period="6mo")
                df = df[
                    (df.index >= pd.Timestamp(start_date)) &
                    (df.index <= pd.Timestamp(end_date))
                ]

            if df is not None and not df.empty:
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                data[ticker] = df
            else:
                data[ticker] = pd.DataFrame()

        except Exception:
            data[ticker] = pd.DataFrame()

    return data


def get_brent_prices(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    try:
        brent = yf.Ticker("BZ=F")
        df = brent.history(start=start_date, end=end_date + timedelta(days=1))

        if df is None or df.empty:
            df = brent.history(period="6mo")
            df = df[
                (df.index >= pd.Timestamp(start_date)) &
                (df.index <= pd.Timestamp(end_date))
            ]

        if df is not None and not df.empty:
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            out = df[["Close"]].rename(columns={"Close": "Brent_Price"})
            out.index.name = "Date"
            return out

    except Exception:
        pass

    return pd.DataFrame()


def get_dollar_rate(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    try:
        start_str = start_date.strftime("%m-%d-%Y")
        end_str   = end_date.strftime("%m-%d-%Y")
        url = (
            "https://olinda.bcb.gov.br/olinda/servico/PTAX/versao/v1/odata/"
            f"CotacaoDolarPeriodo(dataInicial=@dataInicial,dataFinalCotacao=@dataFinalCotacao)?"
            f"@dataInicial='{start_str}'&@dataFinalCotacao='{end_str}'&$top=10000&$format=json"
        )
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json().get("value", [])
        if data:
            df = pd.DataFrame(data)
            df["dataHoraCotacao"] = pd.to_datetime(df["dataHoraCotacao"])
            df = df.sort_values("dataHoraCotacao").set_index("dataHoraCotacao")
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            return df

    except Exception:
        pass

    return pd.DataFrame()


# =========================
# Notícias
# =========================

CATEGORIES = {
    "Governança e Gestão":    ["governança", "administração", "conselho", "assembleia", "gestão", "compliance", "ética", "transparência", "esg"],
    "Decisões Financeiras":   ["lucro", "resultado", "balanço", "receita", "ebitda", "endividamento", "desempenho financeiro", "dividendo", "provento"],
    "Política de Preços":     ["preço", "combustível", "gasolina", "diesel", "gnv", "paridade", "subsídio", "reajuste"],
    "Investimentos":          ["investimento", "parceria", "exploração", "produção", "refino", "petroquímica", "expansão", "projeto"],
    "Operações e Produção":   ["produção", "poço", "campo", "pré-sal", "refinaria", "petroquímica", "óleo", "gás", "bacia", "reserva"],
    "Sustentabilidade":       ["sustentabilidade", "ambiental", "energia renovável", "transição energética", "carbono", "meio ambiente"],
    "Outros":                 [],
}


def _clean_text(txt: Optional[str]) -> str:
    if not txt:
        return ""
    txt = re.sub(r"http\S+", "", txt)
    txt = re.sub(r"[^a-zA-ZÀ-ÿ0-9\s\.,!?:\-\'\"\(\)]", "", txt)
    return txt.strip()


def categorize_news(title: str, content: str) -> str:
    full = f"{title or ''} {content or ''}".lower()
    for cat, keys in CATEGORIES.items():
        if any(k in full for k in keys):
            return cat
    return "Outros"


def get_news(query: str, period: str = "30d", max_results: int = 50, theme: Optional[str] = None) -> List[dict]:
    q = query if not theme or theme == "Todos" else f"{query} {theme}"
    # Limite seguro para evitar timeout no Streamlit Cloud
    safe_max = min(max_results, 30)
    g = GNews(language="pt", country="BR", period=period, max_results=safe_max)
    try:
        news = g.get_news(q) or []
    except Exception:
        news = []

    processed = []
    for n in news:
        title = n.get("title", "")
        desc  = n.get("description", "")
        content = f"{title}. {desc}".strip()
        processed.append({
            "title":          title,
            "description":    desc,
            "content":        content,
            "url":            n.get("url", ""),
            "published_date": n.get("published date", ""),
            "publisher":      (n.get("publisher") or {}).get("title", ""),
        })
    return processed


def fetch_article_body(url: str, max_chars: int = 4000) -> str:
    if not url:
        return ""
    try:
        from newspaper import Article
        art = Article(url, language="pt")
        art.download()
        art.parse()
        txt = (art.text or "").strip()
        if not txt:
            return ""
        return txt[:max_chars] if len(txt) > max_chars else txt
    except Exception:
        return ""


def enrich_news_with_body(news_list: List[dict], use_body: bool = True, max_chars: int = 4000) -> List[dict]:
    if not use_body:
        return [{**n, "body": ""} for n in news_list]
    out = []
    for n in news_list:
        body = fetch_article_body(n.get("url", ""), max_chars=max_chars)
        out.append({**n, "body": body})
    return out


# =========================
# Sentimento em Português
# Prioridade: pysentimiento → VADER → dicionário PT
# =========================

_PYSENTIMIENTO = None
_PYSENTIMIENTO_CHECKED = False

def _ensure_pysentimiento():
    """Tenta carregar o analisador de sentimento em PT (pysentimiento)."""
    global _PYSENTIMIENTO, _PYSENTIMIENTO_CHECKED
    if _PYSENTIMIENTO_CHECKED:
        return _PYSENTIMIENTO
    _PYSENTIMIENTO_CHECKED = True
    try:
        from pysentimiento import create_analyzer
        _PYSENTIMIENTO = create_analyzer(task="sentiment", lang="pt")
    except Exception:
        _PYSENTIMIENTO = None
    return _PYSENTIMIENTO


_VADER = None
_VADER_CHECKED = False

def _ensure_vader():
    """Fallback: VADER em inglês (menos preciso para PT)."""
    global _VADER, _VADER_CHECKED
    if _VADER_CHECKED:
        return _VADER
    _VADER_CHECKED = True
    try:
        import nltk
        from nltk.sentiment import SentimentIntensityAnalyzer
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
        _VADER = SentimentIntensityAnalyzer()
    except Exception:
        _VADER = None
    return _VADER


# Dicionário PT de fallback final
POS_WORDS = {
    "lucro", "crescimento", "recorde", "alta", "positivo", "bom", "excelente",
    "forte", "aumento", "avanço", "aprovação", "investimento", "parceria",
    "descoberta", "eficiência", "produtividade", "otimismo", "valorização",
    "dividendos", "superávit", "expansão", "produção", "recorde",
}
NEG_WORDS = {
    "prejuízo", "queda", "baixa", "negativo", "ruim", "fraco", "problema",
    "dificuldade", "risco", "investigação", "multa", "acidente", "paralisação",
    "dívida", "endividamento", "desvalorização", "incerteza", "subsídio",
    "corrupção", "fraude", "processo", "crise", "perda", "déficit", "redução",
}

# Mapeamento pysentimiento → score numérico
_PYSENTIMIENTO_MAP = {"POS": 1.0, "NEG": -1.0, "NEU": 0.0}


def _sent_score(text: str) -> float:
    """
    Retorna score de sentimento em [-1, 1].
    Ordem de preferência: pysentimiento (PT) → VADER (EN fallback) → dicionário PT.
    """
    if not text:
        return 0.0

    # 1. pysentimiento — melhor para português
    analyzer = _ensure_pysentimiento()
    if analyzer is not None:
        try:
            result = analyzer.predict(text[:512])  # limite do modelo
            label  = result.output  # "POS" | "NEG" | "NEU"
            # Usa a probabilidade da classe vencedora para escalar o score
            prob   = result.probas.get(label, 0.5)
            base   = _PYSENTIMIENTO_MAP.get(label, 0.0)
            # Suaviza: score = direção × (prob - 0.5) × 2   → fica em [-1, 1]
            return float(np.clip(base * (prob - 0.33) * 1.5, -1.0, 1.0))
        except Exception:
            pass

    # 2. VADER (fallback em inglês)
    vader = _ensure_vader()
    if vader is not None:
        try:
            return float(vader.polarity_scores(text)["compound"])
        except Exception:
            pass

    # 3. Dicionário PT simples
    t   = text.lower()
    pos = sum(1 for w in POS_WORDS if w in t)
    neg = sum(1 for w in NEG_WORDS if w in t)
    if pos == 0 and neg == 0:
        return 0.0
    raw = (pos - neg) / max(1, pos + neg)
    return float(np.clip(raw * 0.8, -0.8, 0.8))


def analyze_sentiment_weighted(
    title: str,
    body: str,
    w_title: float = TITLE_WEIGHT_DEFAULT,
    w_body:  float = BODY_WEIGHT_DEFAULT,
) -> Tuple[str, float]:
    t = _clean_text(title or "")
    b = _clean_text(body  or "")
    if not t and not b:
        return "Neutro", 0.0

    score_t = _sent_score(t) if t else 0.0
    score_b = _sent_score(b) if b else 0.0

    # Normaliza pesos para somar 1 (robusto quando body está vazio)
    effective_wb = w_body if b else 0.0
    total_w = w_title + effective_wb
    if total_w == 0:
        score = 0.0
    else:
        score = (w_title * score_t + effective_wb * score_b) / total_w

    label = "Positivo" if score > 0.05 else ("Negativo" if score < -0.05 else "Neutro")
    return label, float(score)


def process_news_with_sentiment(
    news_list: List[dict],
    use_body:  bool  = True,
    w_title:   float = TITLE_WEIGHT_DEFAULT,
    w_body:    float = BODY_WEIGHT_DEFAULT,
) -> List[dict]:
    """
    Enriquece as notícias com sentimento.
    Chamado SEPARADAMENTE do carregamento de dados brutos para permitir
    cache independente no app.py.
    """
    news2 = enrich_news_with_body(news_list, use_body=use_body)
    out = []
    for n in news2:
        title = n.get("title", "")
        body  = n.get("body", "") or n.get("content", n.get("description", "")) or ""
        label, score = analyze_sentiment_weighted(title, body, w_title, w_body)
        out.append({
            **n,
            "body":            body,
            "sentiment":       label,
            "sentiment_score": float(score),
            "category":        categorize_news(title, n.get("description") or ""),
        })
    return out


# =========================
# Transformações / Tabelas
# =========================

def calc_variations(
    stock_data: Dict[str, pd.DataFrame],
    brent_df:   Optional[pd.DataFrame],
    dollar_df:  Optional[pd.DataFrame],
) -> pd.DataFrame:
    variations = pd.DataFrame()
    for tk, df in stock_data.items():
        if df is not None and not df.empty and "Close" in df.columns:
            variations[tk] = df["Close"].pct_change(fill_method=None) * 100.0
    if brent_df is not None and not brent_df.empty and "Brent_Price" in brent_df.columns:
        variations["Brent"] = brent_df["Brent_Price"].pct_change(fill_method=None) * 100.0
    if dollar_df is not None and not dollar_df.empty and "cotacaoVenda" in dollar_df.columns:
        variations["Dólar"] = dollar_df["cotacaoVenda"].pct_change(fill_method=None) * 100.0
    return variations


def normalize_close(df: pd.DataFrame, col: str = "Close") -> pd.Series:
    if df is None or df.empty or col not in df.columns:
        return pd.Series(dtype=float)
    s = df[col].astype(float)
    return (s / s.iloc[0]) * 100.0 if len(s) > 0 and s.iloc[0] != 0 else pd.Series(dtype=float)


def corr_table(series: Dict[str, pd.Series]) -> Tuple[pd.DataFrame, int]:
    """
    Calcula a matriz de correlação descartando NaN (dropna) em vez de
    preencher com zeros, evitando distorção artificial nos resultados.

    Retorna:
        (DataFrame de correlação, n_dias_validos)
    """
    aligned = pd.concat(series, axis=1).dropna()  # ← CORREÇÃO PRINCIPAL
    if aligned.empty:
        return pd.DataFrame(), 0
    n_valid = len(aligned)
    return aligned.corr(), n_valid


def daily_sentiment_series(news_processed: List[dict]) -> pd.Series:
    if not news_processed:
        return pd.Series(dtype=float)

    df = pd.DataFrame(news_processed)
    if "published_date" not in df.columns:
        return pd.Series(dtype=float)

    dt     = pd.to_datetime(df["published_date"], errors="coerce", utc=True)
    dt     = dt.dt.tz_convert("America/Sao_Paulo")
    df["_d"] = dt.dt.date
    df = df.dropna(subset=["_d"])
    if df.empty:
        return pd.Series(dtype=float)

    s = df.groupby("_d")["sentiment_score"].mean()
    s.index = pd.to_datetime(s.index)
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)

    s.name = "Sentiment"
    return s


def build_daily_table(
    stock_data: Dict[str, pd.DataFrame],
    brent_df:   Optional[pd.DataFrame],
    dollar_df:  Optional[pd.DataFrame],
    daily_sent: pd.Series,
) -> pd.DataFrame:
    cols = {}

    if stock_data.get("^BVSP") is not None and not stock_data["^BVSP"].empty:
        cols["IBOV"] = stock_data["^BVSP"]["Close"]
    if stock_data.get("PBR") is not None and not stock_data["PBR"].empty:
        cols["PBR"] = stock_data["PBR"]["Close"]
    if stock_data.get("PBRA") is not None and not stock_data["PBRA"].empty:
        cols["PBRA"] = stock_data["PBRA"]["Close"]
    if stock_data.get("PETR3.SA") is not None and not stock_data["PETR3.SA"].empty:
        cols["PETR3"] = stock_data["PETR3.SA"]["Close"]
    if stock_data.get("PETR4.SA") is not None and not stock_data["PETR4.SA"].empty:
        cols["PETR4"] = stock_data["PETR4.SA"]["Close"]
    if brent_df is not None and not brent_df.empty:
        cols["Brent"] = brent_df["Brent_Price"]
    if dollar_df is not None and not dollar_df.empty and "cotacaoVenda" in dollar_df.columns:
        cols["Dólar"] = dollar_df["cotacaoVenda"]

    df = pd.concat(cols, axis=1) if cols else pd.DataFrame()
    if df.empty:
        return pd.DataFrame()

    df.index = pd.to_datetime(df.index).date
    df = df.groupby(df.index).last()
    df.index = pd.to_datetime(df.index)

    if daily_sent is not None and not daily_sent.empty:
        df = df.join(daily_sent, how="outer")

    return df.sort_index()


def build_variations_table(daily_tbl: pd.DataFrame) -> pd.DataFrame:
    if daily_tbl is None or daily_tbl.empty:
        return pd.DataFrame()
    cols = [c for c in daily_tbl.columns if c != "Sentiment"]
    var  = daily_tbl[cols].pct_change(fill_method=None) * 100.0
    return var.round(3)
