import quandl
from stockstats import StockDataFrame
import pandas as pd
from utils import common


def get_stock(args):
    '''
    create dataset for the required stock
    with the required technical indicators
    '''
    config = common.read_yaml(args.config)
    database = config['database']
    root = config['root']
    t = quandl.get(database, authtoken="qsURFU2ZmHujSwrnY6pk")
    #define the technical indicators
    technical_indicators = ['adj_close','adj_open','volume_delta', 'cr', 'kdjk', 'kdjd', 'kdjj', 'macd', 'macds', 'macdh', 
                            'boll', 'boll_ub', 'boll_lb', 'rsi_6', 'rsi_12', 'wr_10', 'wr_6', 'cci', 
                            'tr', 'atr', 'dma', 'pdi', 'mdi', 'dx', 'adx', 'trix', 'tema', 'vr_10'
                            ]
    #create dataframe with indicators using stockstats
    stocks = StockDataFrame.retype(t)
    #choose the required technical indicators
    indicator = pd.DataFrame(stocks[technical_indicators])
    #convert to standard pandas dataframe
    columns = indicator.columns
    #create dataframe with technical indicators on the selected indicators
    stocks = StockDataFrame.retype(indicator)
    # we want the averages for all the selected technical indicators
    for j in range(len(technical_indicators)//3):
        stocks[[f'{i}_{6+j}_sma' for i in columns]]
        stocks[[f'{i}_{6+j}_mstd' for i in columns]]
        stocks[[f'{i}_{6+j}_ema' for i in columns]]
    stocks.dropna(inplace=True)
    #create csv file
    stocks.to_csv(root)

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="download-stock-database")
    parser.add_argument("--config", required=True, type=str, help="path to config file")
    args = parser.parse_args()

    get_stock(args)
    
