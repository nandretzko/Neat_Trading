"""
Technical indicators as defined in Table 2 of the paper.

All outputs are normalised to roughly [-1, 1] or [0, 1] so the
NEAT network receives well-scaled inputs.

Inputs (column names expected in `df`):
    Open, High, Low, Close, Volume

Outputs (9 columns, matching paper's 11-input vector minus the two
portfolio-state inputs that are injected by TradingEnvironment):
    sma5, sma10, slow_k, slow_d, willr, macd_diff, cci, rsi, adosc
"""
import numpy as np
import pandas as pd


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].values.astype(float)
    high  = df["High"].values.astype(float)
    low   = df["Low"].values.astype(float)
    vol   = df["Volume"].values.astype(float)

    # ------------------------------------------------------------------ #
    # SMA5 / SMA10  — normalised: Ct / SMA(n)                            #
    # ≈ 1 at equilibrium; >1 = price above MA (bullish momentum)         #
    # ------------------------------------------------------------------ #
    sma5  = pd.Series(close).rolling(5).mean().values
    sma10 = pd.Series(close).rolling(10).mean().values
    sma5_n  = close / np.where(sma5  > 0, sma5,  1.0)
    sma10_n = close / np.where(sma10 > 0, sma10, 1.0)

    # ------------------------------------------------------------------ #
    # Stochastic Oscillator (K% and D%)   — [0, 1]                       #
    # ------------------------------------------------------------------ #
    period_k = 14
    ll = pd.Series(low).rolling(period_k).min().values
    hh = pd.Series(high).rolling(period_k).max().values
    rng = np.where((hh - ll) > 1e-8, hh - ll, 1e-8)
    slow_k = (close - ll) / rng          # [0, 1]
    slow_d = pd.Series(slow_k).rolling(3).mean().values

    # ------------------------------------------------------------------ #
    # Williams %R  — [0, 1]  (paper uses positive form: (Hn-Ct)/(Hn-Ln))#
    # ------------------------------------------------------------------ #
    willr = (hh - close) / rng           # [0, 1]

    # ------------------------------------------------------------------ #
    # MACD diff  — Δ(MACD histogram) / Close  (already small)            #
    # ------------------------------------------------------------------ #
    ema12  = pd.Series(close).ewm(span=12, adjust=False).mean().values
    ema26  = pd.Series(close).ewm(span=26, adjust=False).mean().values
    diff   = ema12 - ema26
    signal = pd.Series(diff).ewm(span=9, adjust=False).mean().values
    hist   = diff - signal
    delta_hist   = np.diff(hist, prepend=hist[0])
    macd_diff    = delta_hist / np.where(close > 1e-8, close, 1.0)

    # ------------------------------------------------------------------ #
    # CCI   — clipped to [-3, 3]                                          #
    # ------------------------------------------------------------------ #
    period_cci = 20
    tp     = (high + low + close) / 3.0
    sma_tp = pd.Series(tp).rolling(period_cci).mean().values
    mad    = pd.Series(tp).rolling(period_cci).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True
    ).values
    mad    = np.where(mad > 1e-8, mad, 1e-8)
    cci    = (tp - sma_tp) / (0.015 * mad)
    cci_n  = np.clip(cci / 200.0, -3.0, 3.0)

    # ------------------------------------------------------------------ #
    # RSI   — [0, 1]                                                      #
    # ------------------------------------------------------------------ #
    period_rsi = 14
    delta_c  = np.diff(close, prepend=close[0])
    up_c     = np.where(delta_c > 0, delta_c, 0.0)
    down_c   = np.where(delta_c < 0, -delta_c, 0.0)
    avg_up   = pd.Series(up_c).rolling(period_rsi).mean().values
    avg_down = pd.Series(down_c).rolling(period_rsi).mean().values
    avg_down = np.where(avg_down > 1e-8, avg_down, 1e-8)
    rsi      = 1.0 - 1.0 / (1.0 + avg_up / avg_down)   # [0, 1]

    # ------------------------------------------------------------------ #
    # ADOSC — Chaikin A/D Oscillator                                      #
    # AD(t) = AD(t-1) + CMFV(t)                                          #
    # CMFV (Money Flow Volume) — standard formula:                        #
    #   CLV = ((Close - Low) - (High - Close)) / (High - Low)            #
    #       = (2*Close - High - Low) / (High - Low)                       #
    #   CMFV = CLV * Volume                                               #
    # Oscillator = EMA3(AD) - EMA10(AD), normalised by rolling std       #
    # ------------------------------------------------------------------ #
    hl_rng = np.where((high - low) > 1e-8, high - low, 1e-8)
    clv    = (2.0 * close - high - low) / hl_rng   # Close Location Value ∈ [-1, 1]
    cmfv   = clv * vol
    ad     = np.cumsum(cmfv)
    ema3_ad  = pd.Series(ad).ewm(span=3,  adjust=False).mean().values
    ema10_ad = pd.Series(ad).ewm(span=10, adjust=False).mean().values
    adosc_raw = ema3_ad - ema10_ad
    roll_std  = pd.Series(adosc_raw).rolling(20).std().values
    roll_std  = np.where(roll_std > 1e-8, roll_std, 1e-8)
    adosc_n   = np.clip(adosc_raw / roll_std, -3.0, 3.0)

    return pd.DataFrame({
        "sma5":      sma5_n,
        "sma10":     sma10_n,
        "slow_k":    slow_k,
        "slow_d":    slow_d,
        "willr":     willr,
        "macd_diff": macd_diff,
        "cci":       cci_n,
        "rsi":       rsi,
        "adosc":     adosc_n,
    }, index=df.index)
