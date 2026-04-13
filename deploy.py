"""
日米業種リードラグ戦略 - GitHub Pages デプロイスクリプト
シグナル計算 → index.html 生成 → git commit & push
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.linalg import eigh
from datetime import datetime, timedelta
import json
import subprocess
import os
import warnings

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# パラメータ
# =============================================================================
US_TICKERS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
JP_TICKERS = [f"{i}.T" for i in range(1617, 1634)]

US_NAMES = {
    "XLB": "素材", "XLC": "通信", "XLE": "エネルギー", "XLF": "金融",
    "XLI": "資本財", "XLK": "テクノロジー", "XLP": "生活必需品", "XLRE": "不動産",
    "XLU": "公益", "XLV": "ヘルスケア", "XLY": "一般消費財",
}
JP_NAMES = {
    "1617.T": "食品", "1618.T": "エネルギー資源", "1619.T": "建設・資材",
    "1620.T": "素材・化学", "1621.T": "医薬品", "1622.T": "自動車・輸送機",
    "1623.T": "鉄鋼・非鉄", "1624.T": "機械", "1625.T": "電機・精密",
    "1626.T": "情報通信・サービス", "1627.T": "電力・ガス", "1628.T": "運輸・物流",
    "1629.T": "商社・卸売", "1630.T": "小売", "1631.T": "銀行",
    "1632.T": "金融(除く銀行)", "1633.T": "不動産",
}

US_CYCLICAL = {"XLB", "XLE", "XLF", "XLRE"}
US_DEFENSIVE = {"XLK", "XLP", "XLU", "XLV"}
JP_CYCLICAL = {"1618.T", "1625.T", "1629.T", "1631.T"}
JP_DEFENSIVE = {"1617.T", "1621.T", "1627.T", "1630.T"}

WINDOW = 60
LAMBDA = 0.9
K = 3
Q = 0.3
CFULL_END = "2014-12-31"


# =============================================================================
# シグナル計算
# =============================================================================
def compute_signals():
    all_tickers = US_TICKERS + JP_TICKERS
    end_date = datetime.now() + timedelta(days=1)
    start_date = end_date - timedelta(days=WINDOW * 3)

    print("直近データをダウンロード中...")
    close_frames = []
    for ticker in all_tickers:
        try:
            df = yf.download(ticker, start=start_date.strftime("%Y-%m-%d"),
                           end=end_date.strftime("%Y-%m-%d"),
                           auto_adjust=True, progress=False)
            if len(df) > 0:
                if isinstance(df.columns, pd.MultiIndex):
                    close_frames.append(df[("Close", ticker)].rename(ticker))
                else:
                    close_frames.append(df["Close"].rename(ticker))
        except Exception:
            pass

    close = pd.concat(close_frames, axis=1)
    us_valid = close[US_TICKERS].dropna(how="all").index
    jp_valid = close[JP_TICKERS].dropna(how="all").index
    common_dates = us_valid.intersection(jp_valid)
    close = close.loc[common_dates].ffill().bfill()
    ret_cc = close.pct_change().dropna()

    print("Cfull用データをダウンロード中...")
    close_full_frames = []
    for ticker in all_tickers:
        try:
            df = yf.download(ticker, start="2010-01-01", end=CFULL_END,
                           auto_adjust=True, progress=False)
            if len(df) > 0:
                if isinstance(df.columns, pd.MultiIndex):
                    close_full_frames.append(df[("Close", ticker)].rename(ticker))
                else:
                    close_full_frames.append(df["Close"].rename(ticker))
        except Exception:
            pass

    close_full = pd.concat(close_full_frames, axis=1).ffill().bfill()
    ret_full = close_full.pct_change().dropna()

    # 事前部分空間
    N_U, N_J = len(US_TICKERS), len(JP_TICKERS)
    N = N_U + N_J

    v1 = np.ones(N); v1 /= np.linalg.norm(v1)
    v2 = np.zeros(N); v2[:N_U] = 1.0; v2[N_U:] = -1.0
    v2 -= np.dot(v2, v1) * v1; v2 /= np.linalg.norm(v2)
    v3 = np.zeros(N)
    for i, t in enumerate(all_tickers):
        if t in US_CYCLICAL or t in JP_CYCLICAL: v3[i] = 1.0
        elif t in US_DEFENSIVE or t in JP_DEFENSIVE: v3[i] = -1.0
    v3 -= np.dot(v3, v1) * v1; v3 -= np.dot(v3, v2) * v2; v3 /= np.linalg.norm(v3)
    V0 = np.column_stack([v1, v2, v3])

    # C0
    available = [t for t in all_tickers if t in ret_full.columns]
    ret_sub = ret_full[available]
    mu_f = ret_sub.mean(); sigma_f = ret_sub.std().replace(0, 1e-8)
    z_f = (ret_sub - mu_f) / sigma_f
    corr_sub = z_f.corr()
    C_full = np.zeros((N, N))
    for i, ti in enumerate(all_tickers):
        for j, tj in enumerate(all_tickers):
            if ti in corr_sub.columns and tj in corr_sub.columns:
                val = corr_sub.loc[ti, tj]
                C_full[i, j] = val if not np.isnan(val) else 0.0
    np.fill_diagonal(C_full, 1.0)

    D0 = np.diag(np.diag(V0.T @ C_full @ V0))
    C0_raw = V0 @ D0 @ V0.T
    delta = np.diag(C0_raw); delta = np.where(delta > 0, delta, 1e-8)
    C0 = C0_raw * np.outer(1/np.sqrt(delta), 1/np.sqrt(delta))
    np.fill_diagonal(C0, 1.0)
    C0 = np.nan_to_num(C0, nan=0.0); np.fill_diagonal(C0, 1.0)

    # シグナル
    print("シグナルを計算中...")
    ret_all = ret_cc[all_tickers].values
    T = len(ret_all)
    window = ret_all[T - WINDOW - 1:T - 1]
    mu_w = window.mean(axis=0); sigma_w = window.std(axis=0)
    sigma_w[sigma_w == 0] = 1e-8
    z_window = (window - mu_w) / sigma_w

    Ct = z_window.T @ z_window / WINDOW
    Ct = np.nan_to_num(Ct); np.fill_diagonal(Ct, 1.0); Ct = (Ct + Ct.T) / 2
    C_reg = (1 - LAMBDA) * Ct + LAMBDA * C0
    C_reg = np.nan_to_num(C_reg); np.fill_diagonal(C_reg, 1.0); C_reg = (C_reg + C_reg.T) / 2

    eigenvalues, eigenvectors = eigh(C_reg, subset_by_index=[N - K, N - 1])
    V_K = eigenvectors[:, ::-1]
    V_U, V_J = V_K[:N_U, :], V_K[N_U:, :]

    ret_us_latest = ret_all[-1, :N_U]
    z_U = (ret_us_latest - mu_w[:N_U]) / sigma_w[:N_U]
    ft = V_U.T @ z_U
    z_hat_J = V_J @ ft

    latest_date = ret_cc.index[-1].strftime("%Y-%m-%d")

    # 結果整形
    n_select = max(1, int(len(JP_TICKERS) * Q))
    jp_signals = list(zip(JP_TICKERS, z_hat_J))
    jp_signals.sort(key=lambda x: x[1], reverse=True)
    long_set = {t for t, _ in jp_signals[:n_select]}
    short_set = {t for t, _ in jp_signals[-n_select:]}

    jp_list = []
    for rank, (ticker, sig) in enumerate(jp_signals, 1):
        action = "BUY" if ticker in long_set else ("SELL" if ticker in short_set else "")
        jp_list.append({
            "rank": rank, "ticker": ticker, "name": JP_NAMES[ticker],
            "signal": round(float(sig), 4), "action": action,
        })

    us_sorted = sorted(zip(US_TICKERS, ret_us_latest * 100), key=lambda x: x[1], reverse=True)
    us_list = [{"ticker": t, "name": US_NAMES[t], "return_pct": round(float(r), 2)} for t, r in us_sorted]

    return {
        "us_date": latest_date,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "jp_signals": jp_list,
        "us_returns": us_list,
        "n_long": n_select,
        "n_short": n_select,
    }


# =============================================================================
# HTML生成
# =============================================================================
def generate_html(signal_data):
    data_json = json.dumps(signal_data, ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Lead-Lag Signal</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, 'Segoe UI', 'Hiragino Sans', sans-serif;
       background: #0a0e27; color: #e0e0e0; min-height: 100vh; }}
.container {{ max-width: 600px; margin: 0 auto; padding: 16px; }}
.header {{ text-align: center; padding: 20px 0 12px; }}
.header h1 {{ font-size: 20px; color: #fff; letter-spacing: 1px; }}
.header .sub {{ font-size: 13px; color: #8892b0; margin-top: 4px; }}
.date-box {{ background: #1a1f3d; border-radius: 12px; padding: 14px 16px;
            margin: 12px 0; text-align: center; }}
.date-box .label {{ font-size: 11px; color: #8892b0; text-transform: uppercase; letter-spacing: 1px; }}
.date-box .value {{ font-size: 16px; color: #fff; font-weight: 600; margin-top: 2px; }}
.section-title {{ font-size: 14px; font-weight: 700; color: #8892b0;
                 margin: 20px 0 8px; padding-left: 4px;
                 text-transform: uppercase; letter-spacing: 1px; }}
.signal-card {{ background: #131832; border-radius: 10px; margin: 6px 0;
               padding: 12px 14px; display: flex; align-items: center;
               justify-content: space-between; border-left: 4px solid transparent; }}
.signal-card.buy {{ border-left-color: #00d68f; }}
.signal-card.sell {{ border-left-color: #ff4d6a; }}
.signal-card .left {{ display: flex; align-items: center; gap: 10px; }}
.signal-card .rank {{ font-size: 13px; color: #5a6380; font-weight: 700;
                     width: 24px; text-align: center; }}
.signal-card .info .name {{ font-size: 14px; font-weight: 600; color: #fff; }}
.signal-card .info .ticker {{ font-size: 11px; color: #5a6380; }}
.signal-card .right {{ text-align: right; }}
.signal-card .signal-val {{ font-size: 16px; font-weight: 700; }}
.signal-card .signal-val.pos {{ color: #00d68f; }}
.signal-card .signal-val.neg {{ color: #ff4d6a; }}
.signal-card .badge {{ font-size: 10px; font-weight: 700; padding: 2px 8px;
                      border-radius: 4px; margin-top: 2px; display: inline-block; }}
.badge.buy {{ background: rgba(0,214,143,0.15); color: #00d68f; }}
.badge.sell {{ background: rgba(255,77,106,0.15); color: #ff4d6a; }}
.bar-wrap {{ width: 100%; height: 6px; background: #1e2548; border-radius: 3px;
            margin-top: 6px; position: relative; }}
.bar-fill {{ height: 100%; border-radius: 3px; position: absolute; }}
.bar-fill.pos {{ background: #00d68f; right: 50%; }}
.bar-fill.neg {{ background: #ff4d6a; left: 50%; }}
.bar-center {{ position: absolute; left: 50%; top: 0; width: 1px;
              height: 100%; background: #5a6380; }}
.signal-label {{ font-size: 10px; margin-top: 2px; color: #8892b0; }}
.us-card {{ background: #131832; border-radius: 10px; margin: 4px 0;
           padding: 10px 14px; display: flex; align-items: center;
           justify-content: space-between; }}
.us-card .name {{ font-size: 13px; color: #c0c8e0; }}
.us-card .ticker {{ font-size: 11px; color: #5a6380; }}
.us-card .ret {{ font-size: 14px; font-weight: 700; }}
.us-card .ret.pos {{ color: #00d68f; }}
.us-card .ret.neg {{ color: #ff4d6a; }}
.divider {{ height: 1px; background: #1e2548; margin: 16px 0; }}
.footer {{ text-align: center; padding: 20px 0; font-size: 11px; color: #5a6380; }}
.footer .strategy {{ background: #1a1f3d; border-radius: 8px; padding: 12px;
                    margin: 12px 0; font-size: 12px; color: #8892b0; line-height: 1.6; }}
.tab-bar {{ display: flex; gap: 4px; margin: 12px 0; }}
.tab {{ flex: 1; padding: 10px; text-align: center; font-size: 13px; font-weight: 600;
       background: #131832; border-radius: 8px; cursor: pointer; color: #5a6380; }}
.tab.active {{ background: #1e3a5f; color: #64b5f6; }}
</style>
</head>
<body>
<div class="container" id="app">
  <div class="header">
    <h1>Lead-Lag Signal</h1>
    <div class="sub">日米業種リードラグ戦略</div>
  </div>
  <div id="content"></div>
</div>

<script>
const d = {data_json};

const buyItems = d.jp_signals.filter(s => s.action === 'BUY');
const sellItems = d.jp_signals.filter(s => s.action === 'SELL');

const maxSig = Math.max(...d.jp_signals.map(s => Math.abs(s.signal)), 0.01);

function signalLabel(v) {{
  const a = Math.abs(v);
  if (a >= maxSig * 0.7) return v >= 0 ? '強い買い' : '強い売り';
  if (a >= maxSig * 0.3) return v >= 0 ? '買い' : '売り';
  return '中立';
}}

function signalCard(s) {{
  const cls = s.action === 'BUY' ? 'buy' : (s.action === 'SELL' ? 'sell' : '');
  const valCls = s.signal >= 0 ? 'pos' : 'neg';
  const badge = s.action ? '<div class="badge ' + cls + '">' + (s.action === 'BUY' ? '買い' : '売り') + '</div>' : '';
  const barPct = Math.min(Math.abs(s.signal) / maxSig * 50, 50);
  const barDir = s.signal >= 0 ? 'pos' : 'neg';
  const barStyle = s.signal >= 0
    ? 'width:' + barPct + '%;right:50%'
    : 'width:' + barPct + '%;left:50%';
  const label = signalLabel(s.signal);
  return '<div class="signal-card ' + cls + '">' +
    '<div class="left"><div class="rank">' + s.rank + '</div>' +
    '<div class="info"><div class="name">' + s.name + '</div>' +
    '<div class="ticker">' + s.ticker + '</div></div></div>' +
    '<div class="right" style="min-width:160px">' +
    '<div style="display:flex;align-items:center;justify-content:flex-end;gap:8px">' +
    '<div class="signal-label">' + label + '</div>' +
    '<div class="signal-val ' + valCls + '">' +
    (s.signal >= 0 ? '+' : '') + s.signal.toFixed(4) + '</div>' + badge + '</div>' +
    '<div class="bar-wrap"><div class="bar-center"></div>' +
    '<div class="bar-fill ' + barDir + '" style="' + barStyle + '"></div></div>' +
    '</div></div>';
}}

function showTab(name) {{
  ['signals','us','all'].forEach(t => {{
    document.getElementById('tab-'+t).style.display = t===name ? 'block' : 'none';
  }});
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  event.target.classList.add('active');
}}

document.getElementById('content').innerHTML = `
  <div class="date-box">
    <div class="label">米国市場基準日</div>
    <div class="value">${{d.us_date}}</div>
    <div class="label" style="margin-top:6px">更新: ${{d.updated_at}}</div>
  </div>
  <div class="tab-bar">
    <div class="tab active" onclick="showTab('signals')">シグナル</div>
    <div class="tab" onclick="showTab('us')">米国市場</div>
    <div class="tab" onclick="showTab('all')">全業種</div>
  </div>
  <div id="tab-signals">
    <div class="section-title">買いシグナル (上位${{d.n_long}})</div>
    ${{buyItems.map(s => signalCard(s)).join('')}}
    <div class="divider"></div>
    <div class="section-title">売りシグナル (下位${{d.n_short}})</div>
    ${{sellItems.map(s => signalCard(s)).join('')}}
  </div>
  <div id="tab-us" style="display:none">
    <div class="section-title">米国セクター 当日リターン</div>
    ${{d.us_returns.map(u => '<div class="us-card"><div><div class="name">'+u.name+'</div><div class="ticker">'+u.ticker+'</div></div><div class="ret '+(u.return_pct>=0?'pos':'neg')+'">'+(u.return_pct>=0?'+':'')+u.return_pct.toFixed(2)+'%</div></div>').join('')}}
  </div>
  <div id="tab-all" style="display:none">
    <div class="section-title">全業種シグナル一覧</div>
    ${{d.jp_signals.map(s => signalCard(s)).join('')}}
  </div>
  <div class="footer">
    <div class="strategy">
      上位${{d.n_long}}業種を寄付きでロング<br>
      下位${{d.n_short}}業種を寄付きでショート<br>
      大引けで手仕舞い
    </div>
    Subspace Regularized PCA Lead-Lag Strategy
  </div>
`;
</script>
</body>
</html>"""


# =============================================================================
# デプロイ
# =============================================================================
def deploy():
    print("=" * 50)
    print("  Lead-Lag Signal - GitHub Pages デプロイ")
    print("=" * 50)

    # シグナル計算
    signal_data = compute_signals()
    print(f"\nシグナル計算完了 (基準日: {signal_data['us_date']})")

    # HTML生成
    html_path = os.path.join(SCRIPT_DIR, "index.html")
    html = generate_html(signal_data)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"index.html 生成完了")

    # git commit & push
    print("\nGitHub へ push 中...")
    os.chdir(SCRIPT_DIR)
    subprocess.run(["git", "add", "index.html"], check=True)

    result = subprocess.run(["git", "diff", "--cached", "--quiet"])
    if result.returncode == 0:
        print("変更なし (前回と同じシグナル)")
        return

    date_str = signal_data["us_date"]
    subprocess.run(["git", "commit", "-m", f"update signal: {date_str}"], check=True)
    subprocess.run(["git", "push"], check=True)
    print("\npush 完了!")
    print("GitHub Pages に数分で反映されます")


if __name__ == "__main__":
    import sys
    if "--generate-only" in sys.argv:
        # GitHub Actions用: HTML生成のみ (push はActionsが行う)
        print("=" * 50)
        print("  Lead-Lag Signal - HTML生成 (generate-only)")
        print("=" * 50)
        signal_data = compute_signals()
        print(f"\nシグナル計算完了 (基準日: {signal_data['us_date']})")
        html_path = os.path.join(SCRIPT_DIR, "index.html")
        html = generate_html(signal_data)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        print("index.html 生成完了")
    else:
        # ローカル用: HTML生成 + git push
        deploy()
