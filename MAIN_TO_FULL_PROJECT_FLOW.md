# FULL PROJECT STRUCTURE (TREE TOPOLOGY)

```text
EE4016-Project
|
+-- main.py
+-- get_stock_data.py
+-- train_stock.py
+-- backtest_stock.py
+-- predict_stock.py
+-- run_protocol.py
|
+-- app
|   |
|   +-- menu.py
|   +-- state.py
|   +-- constants.py
|   +-- paths.py
|   |
|   +-- actions
|   |   `-- stock_actions.py
|   |
|   +-- services
|   |   `-- discovery.py
|   |
|   +-- ui
|   |   `-- cli.py
|   |
|   `-- menus
|       +-- data_menu.py
|       +-- train_menu.py
|       +-- backtest_menu.py
|       +-- pipeline_menu.py
|       +-- predict_menu.py
|       `-- results_menu.py
|
+-- historical_data
+-- model
+-- backtest_results
|
+-- MAIN_TO_DATA_DOWNLOAD_FLOW.md
+-- MAIN_TO_DATA_DOWNLOAD_CODE_PATH.md
`-- MAIN_TO_FULL_PROJECT_FLOW.md
```
