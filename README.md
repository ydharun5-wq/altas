# ATLAS V1

ATLAS V1 is a Python project foundation for NSE cash-equity research and
paper trading. It does not connect to a broker, request credentials, or
execute live orders.

## Requirements

- Python 3.12

The current data reader and tests use only the Python standard library.

## Current milestone

`src\atlas\data\ohlcv.py` reads local OHLCV CSV files and rejects malformed
rows, invalid prices, inconsistent OHLC values, negative/non-integer volume,
duplicate bars, and non-increasing timestamps per symbol.

`data\sample_ohlcv.csv` contains the two rows supplied for this project. It is
only a CSV parsing and validation fixture; two bars are not enough to compute
EMA 20/50 or evaluate a trading strategy, and the data has not been verified
as representative historical market data.

## Run tests

From the project root in PowerShell, run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

No package installation is needed for this milestone.

Example:

```powershell
.\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0, 'src'); from atlas.data import read_ohlcv_csv; print(read_ohlcv_csv('data/sample_ohlcv.csv')[0])"
```

Indicators, risk controls, backtesting, persistence, and paper-broker
simulation are not implemented yet.
