# Trading bot configuration
BOT_CONFIG = {
    "symbol": "LTC/USDT",
    "timeframe": "1h",
    "exchange": "binance",
    "api_key": "YOUR_API_KEY",
    "api_secret": "YOUR_API_SECRET",
    "mode": "backtest",  # "backtest" or "live"
}

# Data collection settings
DATA_CONFIG = {
    "historical_days": 365,  # Days of historical data to collect
    "features": ["open", "high", "low", "close", "volume"],
    "technical_indicators": ["rsi", "macd", "bollinger_bands", "ema_9", "ema_21"],
    "data_directory": "historical_data/"
}

# Reinforcement Learning configuration
RL_CONFIG = {
    "algorithm": "dqn",  # "dqn" or "ppo"
    "episode_length": 1000,
    "learning_rate": 0.0001,
    "discount_factor": 0.99,
    "exploration_rate": 0.1,
    "batch_size": 64,
    "memory_size": 100000,
    "update_frequency": 5,
    "model_save_path": "saved_models/"
}

# Risk management settings
RISK_CONFIG = {
    "max_position_size": 0.1,  # Maximum position as percentage of portfolio
    "stop_loss_percentage": 0.03,  # 3% stop loss
    "take_profit_percentage": 0.06,  # 6% take profit
    "max_open_trades": 3,
    "max_daily_loss": 0.05  # 5% daily loss limit
}