import pandas as pd
import numpy as np
import talib
from config.settings import DATA_CONFIG
import logging

class DataPreprocessor:
    def __init__(self):
        self.logger = logging.getLogger('DataPreprocessor')
        
    def add_technical_indicators(self, df):
        """Add technical indicators to the dataframe"""
        self.logger.info("Adding technical indicators")
        
        indicators = DATA_CONFIG['technical_indicators']
        
        # Create a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        if 'rsi' in indicators:
            df['rsi'] = talib.RSI(df['close'], timeperiod=14)
            
        if 'macd' in indicators:
            macd, macd_signal, macd_hist = talib.MACD(
                df['close'], 
                fastperiod=12, 
                slowperiod=26, 
                signalperiod=9
            )
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_hist'] = macd_hist
            
        if 'bollinger_bands' in indicators:
            upper, middle, lower = talib.BBANDS(
                df['close'], 
                timeperiod=20, 
                nbdevup=2, 
                nbdevdn=2
            )
            df['bb_upper'] = upper
            df['bb_middle'] = middle
            df['bb_lower'] = lower
            
        if 'ema_9' in indicators:
            df['ema_9'] = talib.EMA(df['close'], timeperiod=9)
            
        if 'ema_21' in indicators:
            df['ema_21'] = talib.EMA(df['close'], timeperiod=21)

        # Additional potential indicators
        if 'atr' in indicators:
            df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        
        if 'stoch' in indicators:
            df['stoch_k'], df['stoch_d'] = talib.STOCH(
                df['high'], 
                df['low'], 
                df['close']
            )

        if 'obv' in indicators:
            df['obv'] = talib.OBV(df['close'], df['volume'])
            
        # Drop NaN values that occur at the beginning due to indicator calculations
        df.dropna(inplace=True)
        
        return df
    
    def normalize_data(self, df):
        """Normalize data to [0, 1] range for neural network input"""
        self.logger.info("Normalizing data")
        
        # Create a copy
        df_norm = df.copy()
        
        # Store min and max values for later denormalization if needed
        self.min_values = {}
        self.max_values = {}
        
        for column in df_norm.columns:
            min_val = df_norm[column].min()
            max_val = df_norm[column].max()
            
            # Avoid division by zero
            if max_val == min_val:
                df_norm[column] = 0
            else:
                df_norm[column] = (df_norm[column] - min_val) / (max_val - min_val)
            
            self.min_values[column] = min_val
            self.max_values[column] = max_val
        
        return df_norm
    
    def create_sequences(self, df, sequence_length=60):
        """Create input sequences for the neural network"""
        self.logger.info(f"Creating sequences with length {sequence_length}")
        
        X, y = [], []
        
        # Calculate returns for target variable
        df['returns'] = df['close'].pct_change()
        df.dropna(inplace=True)
        
        # Future returns (next candle) as target
        df['target'] = df['returns'].shift(-1)
        df.dropna(inplace=True)
        
        for i in range(len(df) - sequence_length):
            X.append(df.iloc[i:i+sequence_length].values)
            y.append(df.iloc[i+sequence_length]['target'])
            
        return np.array(X), np.array(y)
    
    def preprocess_data(self, df, for_training=True, sequence_length=60):
        """Complete preprocessing pipeline"""
        
        # Add technical indicators
        df = self.add_technical_indicators(df)
        
        # Normalize the data
        df_normalized = self.normalize_data(df)
        
        if for_training:
            # Create sequences for training
            X, y = self.create_sequences(df_normalized, sequence_length)
            return X, y, df_normalized
        else:
            # For real-time data, return the most recent sequence
            if len(df_normalized) >= sequence_length:
                latest_sequence = df_normalized.iloc[-sequence_length:].values.reshape(1, sequence_length, -1)
                return latest_sequence, None, df_normalized
            else:
                self.logger.warning("Not enough data points for a complete sequence")
                return None, None, df_normalized