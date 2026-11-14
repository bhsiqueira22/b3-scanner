"""
B3 Swing Scanner - versao para Render
Servidor Flask que analisa acoes da B3 e entrega um site com o ranking.
"""

import os
import time
import warnings
import pandas as pd
import numpy as np
import yfinance as yf
from flask import Flask, jsonify, render_template_string

warnings.filterwarnings("ignore")

app = Flask(__name__)

UNIVERSO = [
    "PETR4", "VALE3", "ITUB4", "BBAS3", "BBDC4", "WEGE3", "SUZB3", "PRIO3",
    "EGIE3", "RADL3", "LREN3", "CMIG4", "ABEV3", "B3SA3", "ELET3", "GGBR4",
    "CSAN3", "RAIL3", "VIVT3", "KLBN11",
]

# Cache em memoria: evita rebaixar tudo a cada visita
_cache = {"dados": None, "hora": 0}
VALIDADE_CACHE = 1800  # 30 minutos


# ---------------- DADOS ----------------

def baixar(codigo, periodo, intervalo):
    try:
        t = yf.download(codigo + ".SA", period=periodo, interval=intervalo,
                        auto_adjust=True, progress=False, threads=False)
    except Exception:
        return None
    if t is None or t.empty:
        return None
    if isinstance(t.columns, pd.MultiIndex):
        t.columns = t.columns.get_level_values(0)
    cols = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in t.columns for c in cols):
        return None
    t = t[cols].dropna()
    return t if len(t) >= 60 else None


# ---------------- INDICADORES ----------------

def ema(s, p):
    return s.ewm(span=p, adjust=False).mean()


def sma(s, p):
    return s.rolling(p).mean()


def rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(alpha=1 / p, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1 / p, adjust=False).mean()
    return 100 - (100 / (1 + g / l.replace(0, np.nan)))


def atr(t, p=14):
    h, lo, c = t["High"], t["Low"], t["Close"].shift(1)
    tr = pd.concat([h - lo, (h - c).abs(), (lo - c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / p, adjust=False).mean()


def indicadores(t):
    d = t.copy()
    f = d["Close"]
    d["ema9"] = ema(f, 9)
    d["ema21"] = ema(f, 21)
    d["sma50"] = sma(f, 50)
    d["sma200"] = sma(f, 200)
    d["rsi14"] = rsi(f, 14)
    d["atr14"] = atr(d, 14)
    m = ema(f, 12) - ema(f, 26)
    d["macd_hist"] = m - ema(m, 9)
    d["vol_med20"] = d["Volume"].rolling(20).mean()
    d["vol_fin"] = d["Close"] * d["Volume"]
    return d


def tendencia(t):
    if len(t) < 200:
        return "indefinida"
    u = t.iloc[-1]
    if pd.isna(u["sma50"]) or pd.isna(u["sma200"]):
        return "indefinida"
    if u["Close"] > u["sma50"] > u["sma200"]:
        return "alta"
    if u["Close"] < u["sma50"] < u["sma200"]:
        return "baixa"
    return "lateral"


# ---------------- FIBONACCI ----------------

def fibonacci(t, j=5):
    mx, mn = t["High"].values, t["Low"].values
    topos = [i for i in range(j, len(t) - j) if mx[i] == max(mx[i - j:i + j + 1])]
    fundos = [i for i in range(j, len(t) - j) if mn[i] == min(mn[i - j:i + j + 1])]
    if not topos or not fundos:
        return None
    if fundos[-1] < topos[-1]:
        ini = float(t["Low"].iloc[fundos[-1]])
        fim = float(t["High"].iloc[topos[-1]])
    else:
        ini = float(t["High"].iloc[topos[-1]])
        fim = float(t["Low"].iloc[fundos[-1]])
    amp = fim - ini
    if amp == 0:
        return None
    return {f"{n * 100:.1f}%": round(fim - amp * n, 2) for n in [0.382, 0.5, 0.618, 0.786]}


def confluencias(t, fib):
    if not fib:
        return 0
    u = t.iloc[-1]
    medias = [m for m in [u["sma50"], u["sma200"], u["ema21"]] if pd.notna(m)]
    return sum(1 for p in fib.values() if any(abs(p - m) / m <= 0.015 for m in medias))


# ---------------- ANALISE ----------------

def analisar(codigo):
    d = baixar(codigo, "2y", "1d")
    s = baixar(codigo, "5y", "1wk")
    if d is None or s is None:
        return None

    d, s = indicadores(d), indicadores(s)
    u = d.iloc[-1]
    preco = float(u["Close"])
    a = float(u["atr14"]) if pd.notna(u["atr14"]) else preco * 0.03
    stop = round(preco - 2 * a, 2)
    risco = preco - stop
    alvo1 = round(preco + 2 * risco, 2)
    alvo2 = round(preco + 3 * risco, 2)

    fib = fibonacci(d)
    conf = confluencias(d, fib)
    ts, td = tendencia(s), tendencia(d)

    pts = {}
    pts["tendencia"] = {"alta": 12, "lateral": 6, "baixa": 0, "indefinida": 0}[ts] + \
                       {"alta": 8, "lateral": 4, "baixa": 0, "indefinida": 0}[td]
    pts["padrao"] = 0

    vm = u["vol_med20"]
    razao = u["Volume"] / vm if pd.notna(vm) and vm > 0 else 0
    pts["volume"] = 15 if razao >= 2 else 12 if razao >= 1.5 else 8 if razao >= 1.2 else 4 if razao >= 1 else 0
    pts["fibonacci"] = min(conf * 5, 15)

    mom = 0
    r = u["rsi14"]
    if pd.notna(r):
        if 50 <= r <= 65:
            mom += 4
        elif 40 <= r < 50:
            mom += 2
    if pd.notna(u["macd_hist"]) and u["macd_hist"] > 0:
        mom += 3
    if pd.notna(u["ema9"]) and pd.notna(u["ema21"]) and u["ema9"] > u["ema21"]:
        mom += 3
    pts["momentum"] = min(mom, 10)

    rr = (alvo1 - preco) / risco if risco > 0 else 0
    pts["risco_retorno"] = 20 if rr >= 3 else 16 if rr >= 2.5 else 12 if rr >= 2 else 6 if rr >= 1.5 else 0

    score = sum(pts.values())
    imped = []
    if stop >= preco:
        imped.append("stop invalido")
    if rr < 2.0:
        imped.append("R:R abaixo de 2:1")
    if imped:
        score = 0

    cls = ("Excelente" if score >= 85 else "Boa" if score >= 75
           else "Aguardar confirmacao" if score >= 65 else "Nao recomendar")

    return {
        "ticker": codigo,
        "preco": round(preco, 2),
        "score": score,
        "classificacao": cls,
        "tend_semanal": ts,
        "tend_diaria": td,
        "entrada": round(preco, 2),
        "stop": stop,
        "alvo1": alvo1,
        "alvo2": alvo2,
        "rr": round(rr, 2),
        "rsi": round(float(r), 1) if pd.notna(r) else None,
        "atr": round(a, 2),
        "dist_stop": round((preco - stop) / preco * 100, 2),
        "vol_fin": int(u["vol_fin"]) if pd.notna(u["vol_fin"]) else None,
        "confluencias": conf,
        "detalhe": pts,
        "impedimento": "; ".join(imped),
        "data": str(d.index[-1].date()),
    }


def rodar_scanner():
    """Roda a analise de todos os ativos, com cache de 30 minutos."""
    agora = time.time()
    if _cache["dados"] and (agora - _cache["hora"]) < VALIDADE_CACHE:
        return _cache["dados"]

    resultados = []
    for c in UNIVERSO:
        try:
            r = analisar(c)
            if r:
                resultados.append(r)
        except Exception as e:
            print(f"erro em {c}: {e}")
        time.sleep(1.5)

    resultados.sort(key=lambda x: x["score"], reverse=True)
    _cache["dados"] = resultados
    _cache["hora"] = agora
    return resultados


# ---------------- ROTAS ----------------

@app.route("/api/dados")
def api_dados():
    dados = rodar_scanner()
    return jsonify({"ativos": dados, "total": len(dados)})


@app.route("/")
def home():
    return render_template_string(PAGINA)


PAGINA = """
<!DOCTYPE html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>B3 Swing Scanner</title><style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,sans-serif;background:#0d1117;color:#e6edf3;padding:24px}
h1{font-size:26px;margin-bottom:4px}
.sub{color:#8b949e;font-size:13px;margin-bottom:24px}
.kpis{display:flex;gap:16px;margin-bottom:24px;flex-wrap:wrap}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:18px 26px}
.card b{display:block;font-size:30px}
.card small{color:#8b949e;font-size:12px}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:10px;text-align:left;border-bottom:1px solid #21262d}
th{color:#8b949e;font-size:11px;text-transform:uppercase}
tr:hover{background:#161b22}
.s{font-weight:700;font-variant-numeric:tabular-nums}
.e{color:#3fb950}.b{color:#58a6ff}.a{color:#d29922}.n{color:#f85149}
.aviso{background:#1c2128;border-left:3px solid #d29922;padding:12px 16px;
border-radius:4px;font-size:12px;color:#8b949e;margin-bottom:24px}
</style></head><body>
<h1>B3 Swing Scanner</h1>
<p class="sub" id="sub">Carregando...</p>
<div class="aviso">A primeira carga leva de 1 a 3 minutos. Depois disso, os dados ficam em cache por 30 minutos.</div>
<div class="kpis">
<div class="card"><b id="k1">-</b><small>Ativos analisados</small></div>
<div class="card"><b id="k2">-</b><small>Oportunidades (score 65+)</small></div>
</div>
<table><thead><tr>
<th>Ticker</th><th>Preco</th><th>Score</th><th>Status</th><th>Tend. Sem</th>
<th>Tend. Dia</th><th>Entrada</th><th>Stop</th><th>Alvo 1</th><th>R:R</th><th>RSI</th>
</tr></thead><tbody id="tb"></tbody></table>
<script>
fetch('/api/dados').then(r=>r.json()).then(d=>{
  document.getElementById('k1').textContent=d.total;
  document.getElementById('k2').textContent=d.ativos.filter(a=>a.score>=65).length;
  document.getElementById('sub').textContent='Atualizado em '+new Date().toLocaleString('pt-BR');
  const tb=document.getElementById('tb');
  if(!d.ativos.length){
    tb.innerHTML='<tr><td colspan="11">Nenhum ativo retornou dados. O Yahoo pode estar bloqueando o servidor. Tente novamente em alguns minutos.</td></tr>';
    return;
  }
  d.ativos.forEach(a=>{
    const c=a.score>=85?'e':a.score>=75?'b':a.score>=65?'a':'n';
    tb.innerHTML+=`<tr><td><b>${a.ticker}</b></td><td>R$ ${a.preco.toFixed(2)}</td>
    <td class="s ${c}">${a.score}</td><td>${a.classificacao}</td>
    <td>${a.tend_semanal}</td><td>${a.tend_diaria}</td>
    <td>R$ ${a.entrada.toFixed(2)}</td><td>R$ ${a.stop.toFixed(2)}</td>
    <td>R$ ${a.alvo1.toFixed(2)}</td><td>${a.rr}:1</td><td>${a.rsi||'-'}</td></tr>`;
  });
}).catch(e=>{
  document.getElementById('sub').textContent='Erro ao carregar os dados.';
});
</script></body></html>
"""


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
