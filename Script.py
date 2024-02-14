import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import datetime as dt
import sqlite3
from scipy.stats import norm

def download_stock_data(symbol, start, end, database_name='stock_data.db'):
    # Download data
    stock_data = yf.download(symbol, start, end)['Adj Close']
    conn = sqlite3.connect(database_name)
    stock_data.to_sql(symbol, conn, if_exists='replace')
    conn.close()
    return stock_data

def calculate_investment_stats(df, symbol):
    returns = df.pct_change().dropna()
    stats = {
        'mean': round(returns.mean() * 100, 2),
        'std_dev': round(returns.std() * 100, 2),
        'skew': round(returns.skew(), 2),
        'kurt': round(returns.kurtosis(), 2),
        'total_return': round((1 + returns).cumprod().iloc[-1], 4) * 100
    }
    return stats
def calculate_rsi(data, window):
    delta = data.diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    average_gain = up.rolling(window=window).mean()
    average_loss = abs(down.rolling(window=window).mean())
    rs = average_gain / average_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, short_window, long_window):
    short_ema = data.ewm(span=short_window, adjust=False).mean()
    long_ema = data.ewm(span=long_window, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line, signal_line

def calculate_ema(data, periods):
    return data.ewm(ignore_na=False, span=periods, min_periods=periods, adjust=True).mean()
symbol = input('Ingresar un ticker: ')
start_date = input('Fecha de inicial(YYYY-MM-DD): ')
end_date = input('Fecha final(YYYY-MM-DD): ')
start = dt.datetime.strptime(start_date, '%Y-%m-%d')
end = dt.datetime.strptime(end_date, '%Y-%m-%d')

if end > dt.datetime.now():
    print("La fecha final debe ser menor a la fecha de hoy, se utilizará la fecha actual")
else:
    df = download_stock_data(symbol, start, end)

df = download_stock_data(symbol, start, end)
stats = calculate_investment_stats(df, symbol)
print(f"\nStatistics:\nMean: {stats['mean']}%\nStd. Dev: {stats['std_dev']}%\nSkew: {stats['skew']}\nKurt: {stats['kurt']}\nTotal Return: {stats['total_return']}%")

# Calculate EMAs
ema_data = pd.DataFrame(index=df.index)
for n in [50, 200]:
    ema_data[f"EMA_{n}"] = calculate_ema(df, n)
rsi = calculate_rsi(df, 14)
macd_line, signal_line = calculate_macd(df, 12, 26)
fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(18,18))
ax1.plot(df, label="Close")
for n in [50, 200]:
    ax1.plot(ema_data[f"EMA_{n}"], label=f"EMA_{n}")
    
ax1.set_ylabel("Price")
ax1.set_title("Stock Closing Price and EMAs")
ax1.legend(loc="best")

ax2.plot(rsi, label="RSI", color='orange')
ax2.axhline(0, linestyle='--', alpha=0.1)
ax2.axhline(20, linestyle='--', alpha=0.5)
ax2.axhline(30, linestyle='--')
ax2.axhline(70, linestyle='--')
ax2.axhline(80, linestyle='--', alpha=0.5)
ax2.axhline(100, linestyle='--', alpha=0.1)
ax2.set_ylabel("RSI")
ax2.legend(loc="best")

ax3.plot(macd_line, label='MACD Line', color='blue')
ax3.plot(signal_line, label='Signal Line', color='red')
ax3.axhline(0, color='black', linewidth=0.5)
ax3.set_ylabel("MACD")
ax3.legend(loc="best")
ax3.set_xlabel("Fecha")

plt.show()
daily_returns = df.pct_change().dropna()
mu, std = norm.fit(daily_returns)
daily_returns.hist(bins=50, figsize=(14, 7), grid=True, alpha=0.75, density=True)
plt.plot(np.linspace(*plt.xlim(), 100), norm.pdf(np.linspace(*plt.xlim(), 100), mu, std), 'r', linewidth=2)
plt.title("Histograma " f'{symbol} Retornos diarios')
plt.xlabel('Retornos diarios')
plt.ylabel('Frecuencia')
plt.legend(['Distribución Normal'])
plt.show()

def monte_carlo_var_cvar(returns, num_simulations, confidence_level):
    mean = np.mean(returns)
    std_dev = np.std(returns)    
    simulated_returns = np.random.normal(mean, std_dev, num_simulations)
    var = np.percentile(simulated_returns, 100 * (1 - confidence_level))
    cvar = simulated_returns[simulated_returns < var].mean()
    return var, cvar
num_simulations = 1000
confidence_level = 0.95
var, cvar = monte_carlo_var_cvar(daily_returns, num_simulations, confidence_level)

#Pongo las métricas en una figura
fig, ax = plt.subplots(figsize=(10, 8))
ax.text(0.1, 0.9, f"Métricas: {symbol} ({start_date} - {end_date})", fontsize=12, weight='bold')
ax.text(0.1, 0.8, f"Retorno diario Promedio: {stats['mean']}%", fontsize=10)
ax.text(0.1, 0.7, f"Desvío Estándar: {stats['std_dev']}%", fontsize=10)
ax.text(0.1, 0.6, f"Curtósis: {stats['kurt']}", fontsize=10)
ax.text(0.1, 0.5, f"Retorno total: {stats['total_return']}%", fontsize=10)
ax.text(0.1, 0.4, f"Valor en riesgo al {confidence_level*100}% : {var*100:.2f}%", fontsize=10)
ax.text(0.1, 0.3, f"Valor en riesgo condicional al {confidence_level*100}%: {cvar*100:.2f}%", fontsize=10)
ax.axis('off')
plt.savefig("flyer.png", dpi=300, bbox_inches='tight')
plt.show()