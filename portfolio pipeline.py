import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from numba import jit

# ==============================================================================
# 1. THE BRAIN & ENGINES (CORE MATH)
# ==============================================================================
@jit(nopython=True)
def calculate_hurst(price_array: np.ndarray, max_lag: int = 20) -> float:
    lags = np.arange(2.0, float(max_lag + 1))
    variances = np.zeros(len(lags))
    for i in range(len(lags)):
        lag = int(lags[i])
        diffs = price_array[lag:] - price_array[:-lag]
        variances[i] = np.var(diffs)
    x = np.log(lags)
    y = np.log(variances)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    covariance = np.sum((x - x_mean) * (y - y_mean))
    variance = np.sum((x - x_mean)**2)
    return (covariance / variance) / 2.0

def apply_rolling_regime_filter(df: pd.DataFrame, window: int = 300, max_lag: int = 20):
    prices = df['close'].values
    hurst_series = np.full(len(prices), np.nan)
    for i in range(window, len(prices)):
        window_prices = prices[i-window:i]
        hurst_series[i] = calculate_hurst(window_prices, max_lag)
    df['hurst'] = hurst_series
    df['regime'] = 0 
    df.loc[df['hurst'] > 0.55, 'regime'] = 1  
    return df

def apply_session_filter(df: pd.DataFrame):
    df['hour'] = df['datetime'].dt.hour
    df['session'] = 0 
    # High Liquidity Entry Window (09:00 to 17:59 Server Time)
    df.loc[(df['hour'] >= 9) & (df['hour'] < 18), 'session'] = 2
    return df

@jit(nopython=True)
def generate_trend_signals(prices: np.ndarray, spread_pct: float, mom_lookback: int = 50, vol_lookback: int = 48, stop_mult: float = 4.0, floor_mult: float = 20.0):
    n = len(prices)
    signals = np.zeros(n)
    stops = np.full(n, np.nan)
    min_stop_distance = spread_pct * floor_mult
    
    for i in range(max(mom_lookback, vol_lookback), n):
        window_prices = prices[i-vol_lookback:i]
        current_vol = np.std(window_prices)
        momentum = prices[i] - prices[i-mom_lookback]
        stop_distance = max(stop_mult * current_vol, min_stop_distance)
        
        if momentum > 0:
            signals[i] = 1
            stops[i] = np.max(window_prices) - stop_distance
        elif momentum < 0:
            signals[i] = -1
            stops[i] = np.min(window_prices) + stop_distance
            
    return signals, stops

def apply_trend_engine(df: pd.DataFrame, spread_bps: float):
    signals, stops = generate_trend_signals(df['close'].values, spread_bps / 10000)
    df['trend_signal'] = signals
    df['trend_stop'] = stops
    return df

@jit(nopython=True)
def simulate_trend_execution(prices, regimes, sessions, trend_sig, trend_stops, spread_pct, commission_pct):
    n = len(prices)
    position = 0 
    current_stop = np.nan
    equity_curve = np.ones(n)
    trade_logs = np.zeros(n) 
    locked_direction = 0 
    
    for i in range(1, n):
        equity_curve[i] = equity_curve[i-1]
        
        if locked_direction == 1 and trend_sig[i] <= 0: locked_direction = 0
        elif locked_direction == -1 and trend_sig[i] >= 0: locked_direction = 0
            
        if position != 0:
            bar_return = (prices[i] - prices[i-1]) / prices[i-1]
            equity_curve[i] = equity_curve[i-1] * (1 + (position * bar_return))
            
            exit_triggered = False
            if position == 1:
                if trend_stops[i] > current_stop: current_stop = trend_stops[i]
                if prices[i] <= current_stop: exit_triggered = True
            elif position == -1:
                if trend_stops[i] < current_stop or np.isnan(current_stop): current_stop = trend_stops[i]
                if prices[i] >= current_stop: exit_triggered = True
                        
            if exit_triggered:
                equity_curve[i] = equity_curve[i] * (1 - spread_pct - commission_pct)
                trade_logs[i] = 2 if position == 1 else -2
                locked_direction = position 
                position = 0
                continue 
                
        if position == 0:
            if regimes[i] == 1 and sessions[i] == 2 and trend_sig[i] != 0:
                if trend_sig[i] != locked_direction: 
                    position = trend_sig[i]
                    current_stop = trend_stops[i]
                    trade_logs[i] = position
                    equity_curve[i] = equity_curve[i] * (1 - spread_pct - commission_pct)

    return equity_curve, trade_logs

# ==============================================================================
# 2. METRICS PARSER
# ==============================================================================
def calculate_metrics(equity_series, periods_per_year=252*24):
    returns = equity_series.pct_change().dropna()
    if len(returns) == 0 or returns.std() == 0: return {}
    total_return = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1
    ann_vol = returns.std() * np.sqrt(periods_per_year)
    days = len(equity_series) / (periods_per_year / 252)
    ann_return = (1 + total_return) ** (365.25 / days) - 1 if days > 0 else 0
    sharpe = ann_return / ann_vol if ann_vol != 0 else 0
    downside_vol = returns[returns < 0].std() * np.sqrt(periods_per_year)
    sortino = ann_return / downside_vol if downside_vol != 0 else 0
    max_dd = ((equity_series - equity_series.cummax()) / equity_series.cummax()).min()
    return {
        "Total Return": f"{total_return * 100:.2f}%", "Ann. Return": f"{ann_return * 100:.2f}%",
        "Sharpe Ratio": f"{sharpe:.3f}", "Sortino Ratio": f"{sortino:.3f}", "Max DD": f"{max_dd * 100:.2f}%"
    }

def calculate_trade_stats(df, pip_mult):
    trades = []
    entry_price = 0
    entry_idx = 0
    position = 0
    
    for i in range(len(df)):
        log = df['trade_log'].iloc[i]
        if log == 1 or log == -1: 
            position = log
            entry_price = df['close'].iloc[i]
            entry_idx = i
        elif log == 2 or log == -2: 
            exit_price = df['close'].iloc[i]
            bars_held = i - entry_idx
            pip_diff = (exit_price - entry_price) * pip_mult * position
            trades.append({'pips': pip_diff, 'bars_held': bars_held})
            position = 0
            
    if not trades: return {"Total Trades": 0}
        
    pips = [t['pips'] for t in trades]
    wins = [p for p in pips if p > 0]
    losses = [p for p in pips if p <= 0]
    
    return {
        "Total Trades": len(trades),
        "Win Rate": f"{(len(wins) / len(trades)) * 100:.2f}%",
        "Avg Trade (Pips)": f"{np.mean(pips):.2f}",
        "Avg Win (Pips)": f"{np.mean(wins):.2f}" if wins else "0",
        "Avg Loss (Pips)": f"{np.mean(losses):.2f}" if losses else "0"
    }

# ==============================================================================
# 3. MASTER PORTFOLIO EXECUTION
# ==============================================================================
if __name__ == "__main__":
    print("[SYSTEM] Booting Multi-Asset Portfolio Architecture...")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # The Asset Dictionary
    portfolio = {
        'EURUSD': {'file': 'EURUSD_Cleaned.csv', 'pip_mult': 10000},
        'USDJPY': {'file': 'USDJPY_Cleaned.csv', 'pip_mult': 100},
        'XAUUSD': {'file': 'XAUUSD_Cleaned.csv', 'pip_mult': 10}
    }
    
    spread_bps = 1.5
    commission_bps = 0.5
    
    # Dictionary to hold the hourly returns of each asset to build the Master Fund
    portfolio_returns = {}

    for asset, config in portfolio.items():
        csv_path = os.path.join(current_dir, config['file'])
        
        try:
            df = pd.read_csv(csv_path)
            df['datetime'] = pd.to_datetime(df['datetime'])
        except FileNotFoundError:
            print(f"\n[WARNING] {asset} data file not found. Skipping...")
            continue
            
        print(f"\n[SYSTEM] Compiling Brain & Engine for {asset}...")
        df = apply_rolling_regime_filter(df, window=300)
        df = apply_session_filter(df)
        df = apply_trend_engine(df, spread_bps=spread_bps)
        
        df_tradable = df.dropna().reset_index(drop=True)
        
        equity, logs = simulate_trend_execution(
            df_tradable['close'].values, df_tradable['regime'].values, df_tradable['session'].values, 
            df_tradable['trend_signal'].values, df_tradable['trend_stop'].values, 
            spread_bps / 10000, commission_bps / 10000
        )
        
        df_tradable['equity'] = equity
        df_tradable['trade_log'] = logs
        
        # Calculate hourly returns for this asset
        df_tradable['hourly_return'] = df_tradable['equity'].pct_change().fillna(0)
        
        # Store for the Master Fund calculation (indexed by datetime for perfect alignment)
        portfolio_returns[asset] = df_tradable.set_index('datetime')['hourly_return']
        
        # Print Individual Asset Performance
        print(f"--- {asset} ISOLATED PERFORMANCE ---")
        metrics = calculate_metrics(df_tradable['equity'])
        stats = calculate_trade_stats(df_tradable, config['pip_mult'])
        print(f"Sharpe Ratio: {metrics.get('Sharpe Ratio', 'N/A')} | Total Return: {metrics.get('Total Return', 'N/A')}")
        print(f"Win Rate: {stats.get('Win Rate', 'N/A')} | Avg Win: {stats.get('Avg Win (Pips)', 'N/A')} pips | Avg Loss: {stats.get('Avg Loss (Pips)', 'N/A')} pips")

    # ==========================================================================
    # 4. MASTER FUND CONSOLIDATION
    # ==========================================================================
    if portfolio_returns:
        print("\n[SYSTEM] Aligning time-series arrays and calculating Master Portfolio...")
        
        # Merge all asset returns on their exact datetimes (handles missing ticks/holidays perfectly)
        master_df = pd.DataFrame(portfolio_returns).fillna(0)
        
        # The Master Fund return is the average return of the active systems per hour (Equal Weighting)
        master_df['portfolio_return'] = master_df.mean(axis=1)
        
        # Rebuild the Master Equity Curve
        master_df['master_equity'] = (1 + master_df['portfolio_return']).cumprod()
        
        print("\n================================================")
        print("   MASTER FUND PERFORMANCE (EQUAL WEIGHTED)   ")
        print("================================================")
        master_metrics = calculate_metrics(master_df['master_equity'])
        for k, v in master_metrics.items(): 
            print(f"{k}: {v}")
        print("================================================\n")
