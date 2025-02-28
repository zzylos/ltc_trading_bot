import os
import pandas as pd
import numpy as np
import ccxt
from datetime import datetime, timedelta
import time
from config.settings import BOT_CONFIG, DATA_CONFIG
import logging

class DataCollector:
    def __init__(self):
        self.exchange_id = BOT_CONFIG["exchange"]
        self.symbol = BOT_CONFIG["symbol"]
        self.timeframe = BOT_CONFIG["timeframe"]
        self.exchange = getattr(ccxt, self.exchange_id)({
            'apiKey': BOT_CONFIG['api_key'],
            'secret': BOT_CONFIG['api_secret'],
            'enableRateLimit': True,
        })
        
        self.data_dir = DATA_CONFIG["data_directory"]
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
        self.logger = logging.getLogger('DataCollector')
        
    def fetch_historical_data(self):
        """Fetch historical OHLCV data from the exchange"""
        self.logger.info(f"Fetching historical data for {self.symbol}")
        
        # Calculate timespan
        end_date = datetime.now()
        start_date = end_date - timedelta(days=DATA_CONFIG["historical_days"])
        
        # Convert dates to milliseconds timestamp
        since = int(start_date.timestamp() * 1000)
        
        all_candles = []
        
        # Fetch data in chunks to avoid exchange limits
        while since < end_date.timestamp() * 1000:
            self.logger.debug(f"Fetching chunk from {datetime.fromtimestamp(since/1000)}")
            try:
                candles = self.exchange.fetch_ohlcv(
                    symbol=self.symbol,
                    timeframe=self.timeframe,
                    since=since,
                    limit=1000  # Most exchanges limit to 1000 candles per request
                )
                
                if not candles:
                    break
                
                all_candles.extend(candles)
                
                # Update the since parameter for next iteration
                since = candles[-1][0] + 1
                
                # Respect rate limits
                time.sleep(self.exchange.rateLimit / 1000)
                
            except ccxt.NetworkError as e:
                self.logger.error(f"Network error: {str(e)}")
                time.sleep(5)
            except ccxt.ExchangeError as e:
                self.logger.error(f"Exchange error: {str(e)}")
                break
            except Exception as e:
                self.logger.error(f"Unexpected error: {str(e)}")
                break
        
        # Convert to DataFrame
        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Save to file
        filename = f"{self.data_dir}{self.symbol.replace('/', '_')}_{self.timeframe}.csv"
        df.to_csv(filename)
        self.logger.info(f"Saved {len(df)} candles to {filename}")
        
        return df
    
    def fetch_latest_data(self, limit=100):
        """Fetch the most recent candles for live trading"""
        try:
            candles = self.exchange.fetch_ohlcv(
                symbol=self.symbol,
                timeframe=self.timeframe,
                limit=limit
            )
            
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching latest data: {str(e)}")
            return None