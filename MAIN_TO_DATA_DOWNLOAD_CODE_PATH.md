# MAIN -> DOWNLOAD (ASCII CODE MAP)

```text
                           ┌──────────────────────┐
                           │       main.py        │
                           │  main()              │
                           └──────────┬───────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────┐
                    │ app/menu.py                      │
                    │ interactive_menu()               │
                    │ MENU_ACTIONS["1"] -> run_data...│
                    └────────────────┬─────────────────┘
                                     │
                                     ▼
                   ┌──────────────────────────────────┐
                   │ app/menus/data_menu.py           │
                   │ run_data_menu()                  │
                   └──────────────┬───────────────────┘
                                  │
                                  ▼
                  ┌──────────────────────────────────┐
                  │ manual parameter path            │
                  │ tickers/interval/historical range│
                  └──────────────┬───────────────────┘
                                 │
                                 ▼
       ┌────────────────────────────────────────────────────────────┐
       │ app/actions/stock_actions.py                              │
       │ - run_script(...)                                         │
       └──────────────┬─────────────────────────────────────────────┘
                       │
                       ▼
        ┌────────────────────────────────────────────────────────────┐
        │ subprocess.run([sys.executable, "get_stock_data.py", ...])│
        └──────────────┬─────────────────────────────────────────────┘
                       │
                       ▼
        ┌────────────────────────────────────────────────────────────┐
        │ get_stock_data.py                                         │
        │ parse args -> validate -> boundary adjust -> yfinance     │
        │ normalize OHLCV -> save CSV                               │
        └──────────────┬─────────────────────────────────────────────┘
                       │
                       ▼
                historical_data/<TICKER>_<RANGE>_<INTERVAL>.csv
```

```text
DEPENDENCY / SUPPORT MODULE GRAPH

app/menus/data_menu.py
  ├── app/ui/cli.py
  │     ├── ask()
  │     ├── ask_yes_no()
  │     ├── clear_screen()
  │     └── pause()
  ├── app/constants.py
  │     └── INTERVAL_LIMITS
  ├── app/services/discovery.py
  │     ├── duration_to_years()
  │     └── max_lookback_label()
  └── app/actions/stock_actions.py
       └── run_script()

app/services/discovery.py
  ├── app/state.py
  │     ├── load_state()
  │     └── save_state()   # no auto-create .menu_state.json
  └── app/paths.py
        ├── PROJECT_ROOT
        └── STATE_PATH
```
