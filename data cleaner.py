# USE THIS IN CASE YOU WANT TO MANUALLY INSTALL THE DATA FROM QDM OTHERWISE YOU CAN IGNORE THE CODE AS I PROVIDE THE CLEANED CSV IN THE REPO


import MetaTrader5 as mt5
import pandas as pd
import os

print("[SYSTEM] Booting MT5 Live Data Ingestion Node...")

# 1. Establish connection to the MetaTrader 5 terminal
if not mt5.initialize():
    print("[FATAL ERROR] initialize() failed. Ensure MT5 terminal is open.")
    mt5.shutdown()
    quit()

# 2. Define Parameters
symbol = "USDJPY_QDM"
timeframe = mt5.TIMEFRAME_H1
number_of_bars = 100000  # 100,000 hours is roughly 11.5 years of data. 

print(f"[SYSTEM] Requesting the last {number_of_bars} hours of {symbol}...")

# 3. Fetch the data directly from the broker's server
rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, number_of_bars)

if rates is None:
    print(f"[FATAL ERROR] Failed to fetch data. Is {symbol} visible in your MT5 Market Watch?")
    mt5.shutdown()
    quit()

# 4. Shut down the connection immediately to preserve sterility
mt5.shutdown()
print("[SYSTEM] Data received. Connection severed.")

# 5. Format into the exact Institutional Master Pipeline structure
df = pd.DataFrame(rates)

# MT5 provides time as a UNIX timestamp in seconds. Convert it to readable datetime.
df['datetime'] = pd.to_datetime(df['time'], unit='s')

# Isolate only the two columns the Master Architecture needs
df_clean = df[['datetime', 'close']]

# 6. Save to disk in the current working directory
current_dir = os.path.dirname(os.path.abspath(__file__))
filename ="USDJPY_Cleaned.csv"
file_path = os.path.join(current_dir, filename)

df_clean.to_csv(file_path, index=False)

print(f"\n[SUCCESS] Extracted {len(df_clean)} rows of perfect algorithmic data.")
print(f"[SUCCESS] File written to: {file_path}")
print("[ACTION] You may now run your Master Quant Pipeline.")
