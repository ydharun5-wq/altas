# ATLAS development rules

- Paper trading only. Never add live-order execution or request broker credentials.
- Keep risk checks deterministic and fail closed when inputs are missing or invalid.
- AI components may propose changes but must not bypass risk controls or deploy unvalidated strategies.
- Use local or explicitly authorized historical data; label synthetic data clearly.
- Run the focused test suite after changing code: `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`.
