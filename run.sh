#hrun -c 12 -m 16 python src/train.py --config configs/config --parallel 1 --data_path data/Combined_News_SPY.csv
#hrun  -m 30 -G python src/train.py --config configs/mlp_configs --parallel -1 --data_path data/Combined_News_SPY.csv
#hrun -N s04 -c 12 -m 100 python src/train.py --config configs/ticker_config --parallel 0 --data_path data/Combined_News_Ticker.csv
hrun  -m 30 -G python src/train.py --config configs/mlp_ticker_configs --parallel 0 --data_path ./data/Combined_News_Ticker.csv
#hrun -N s04 -c 12 -m 100 python src/train.py --config configs/ticker_config --parallel 0 --data_path data/Combined_News_Ticker_10.csv
#hrun  -m 30 -G python src/train.py --config configs/mlp_ticker_configs --parallel 0 --data_path ./data/Combined_News_Ticker_10.csv
#hrun  -m 30 -G python src/train.py --config configs/test --parallel 0 --data_path ./data/Combined_News_Ticker_10.csv
