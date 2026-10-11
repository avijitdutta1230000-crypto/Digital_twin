
### traffic (synthetic), 6-h horizon

| Model | MAE | RMSE | MAPE % |
|---|---|---|---|
| Persistence | 8.605 | 11.988 | 34.69 |
| LSTM | 2.103 | 3.966 | 11.25 |
| CNN-LSTM | 2.063 | 3.921 | 10.90 |
| MM-STFormer (ours) | 2.182 | 3.961 | 11.28 |
|   w/o exogenous fusion | 2.561 | 4.542 | 13.01 |
|   w/o graph mixing | 2.097 | 3.859 | 11.07 |

### pollution (synthetic), 6-h horizon

| Model | MAE | RMSE | MAPE % |
|---|---|---|---|
| Persistence | 11.301 | 14.890 | 21.52 |
| LSTM | 4.471 | 6.679 | 9.98 |
| CNN-LSTM | 4.513 | 6.811 | 10.12 |
| MM-STFormer (ours) | 4.352 | 6.964 | 9.51 |
|   w/o exogenous fusion | 6.142 | 9.521 | 12.85 |
|   w/o graph mixing | 4.330 | 6.911 | 9.47 |

### electricity (synthetic), 6-h horizon

| Model | MAE | RMSE | MAPE % |
|---|---|---|---|
| Persistence | 2.122 | 2.674 | 35.33 |
| LSTM | 0.424 | 0.847 | 9.31 |
| CNN-LSTM | 0.413 | 0.813 | 8.84 |
| MM-STFormer (ours) | 0.438 | 0.866 | 9.26 |
|   w/o exogenous fusion | 0.480 | 0.907 | 9.93 |
|   w/o graph mixing | 0.437 | 0.867 | 9.31 |
