# Litecoin (LTC) Trading Bot with Reinforcement Learning

A sophisticated cryptocurrency trading bot for Litecoin that uses reinforcement learning to make trading decisions. The bot analyzes historical data, technical indicators, and market patterns to develop profitable trading strategies that improve over time.

## Features

- **Data Collection**: Automatically fetches and processes historical LTC price data
- **Reinforcement Learning**: Implements both DQN and PPO algorithms for adaptive trading
- **Technical Analysis**: Calculates multiple technical indicators to inform trading decisions
- **Risk Management**: Built-in stop-loss, take-profit, and position sizing strategies
- **Backtesting**: Comprehensive backtesting framework to evaluate strategies
- **Live Trading**: Full integration with exchange APIs for automated trading
- **Performance Analytics**: Detailed performance metrics and visualizations

## Installation

Clone the repository:
```bash
git clone https://github.com/zzylos/ltc-trading-bot.git
cd ltc-trading-bot
pip install -r requirements.txt
```
Note: TA-Lib installation may require additional steps:

For Windows: Download and install from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib

For Mac: brew install ta-lib

For Linux: apt-get install ta-lib

## Configuration

Edit the config/settings.py file to set up your:

Exchange API keys

Risk parameters

Model configuration

Data collection settings

## Usage

Data Collection:
```bash
python main.py --mode train
```
Model Training:
```bash
python main.py --mode train --model dqn --episodes 100
```
or
```bash
python main.py --mode train --model ppo --episodes 100
```
Backtesting:
```bash
python main.py --mode backtest --model dqn --load_model saved_models/dqn_model_ep100.h5
```
or
```bash
python main.py --mode backtest --model ppo --load_model saved_models/ppo_model_ep100
```
Live Training (dangerous)
```bash
python main.py --mode live --model dqn --load_model saved_models/dqn_model_ep100.h5
```
or
```bash
python main.py --mode live --model ppo --load_model saved_models/ppo_model_ep100
```

## Risk Warning
This software is for educational purposes only. Cryptocurrency trading carries significant risk. Always test thoroughly with small amounts before committing significant capital. The creators are not responsible for any financial losses incurred using this software.

(Note from a human here, this was entirely created by Claude 3.7 Sonnet as a demonstration of its coding abilities, this was CtrlC and CtrlV'd straight from Claude, please only run this with historical data.)
