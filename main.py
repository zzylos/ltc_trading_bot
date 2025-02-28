import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
import time
import logging

# Import project modules
from config.settings import BOT_CONFIG, RL_CONFIG, DATA_CONFIG
from data.data_collector import DataCollector
from data.data_preprocessor import DataPreprocessor
from models.dqn_agent import DQNAgent
from models.ppo_agent import PPOAgent
from risk_management.risk_manager import RiskManager
from exchange.exchange_api import ExchangeAPI
from backtesting.simulator import BacktestSimulator
from utils.logger import setup_logger
from utils.metrics import PerformanceAnalyzer

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Litecoin Trading Bot with Reinforcement Learning')
    parser.add_argument('--mode', choices=['train', 'backtest', 'live'], default='backtest',
                        help='Operation mode: train, backtest, or live trading')
    parser.add_argument('--model', choices=['dqn', 'ppo'], default=RL_CONFIG['algorithm'],
                        help='Reinforcement learning algorithm to use')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Path to a pre-trained model to load')
    parser.add_argument('--episodes', type=int, default=50,
                        help='Number of episodes for training')
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logger()
    logger.info(f"Starting LTC Trading Bot in {args.mode} mode")
    
    # Create data collector and preprocessor
    data_collector = DataCollector()
    data_preprocessor = DataPreprocessor()
    
    # Load or fetch historical data
    data_filename = f"{DATA_CONFIG['data_directory']}{BOT_CONFIG['symbol'].replace('/', '_')}_{BOT_CONFIG['timeframe']}.csv"
    
    if os.path.exists(data_filename) and args.mode != 'live':
        logger.info(f"Loading historical data from {data_filename}")
        data = pd.read_csv(data_filename, index_col='timestamp', parse_dates=True)
    else:
        logger.info("Fetching historical data")
        data = data_collector.fetch_historical_data()
    
    # Preprocess data
    logger.info("Preprocessing data")
    sequence_length = 60  # Number of time steps to include in each state
    
    # Add technical indicators and normalize
    X, y, preprocessed_data = data_preprocessor.preprocess_data(data, for_training=True, sequence_length=sequence_length)
    
    # Determine state and action space dimensions
    state_size = X.shape[1:]  # (sequence_length, num_features)
    action_size = 3  # Buy, Sell, Hold
    
    # Initialize the model
    if args.model == 'dqn':
        model = DQNAgent(state_size=state_size, action_size=action_size)
        logger.info("DQN agent initialized")
    else:
        model = PPOAgent(state_size=state_size, action_size=action_size)
        logger.info("PPO agent initialized")
    
    # Load pre-trained model if specified
    if args.load_model:
        logger.info(f"Loading pre-trained model from {args.load_model}")
        if args.model == 'dqn':
            model.load(args.load_model)
        else:
            model.load(f"{args.load_model}_actor", f"{args.load_model}_critic")
    
    # Training mode
    if args.mode == 'train':
        logger.info("Starting training mode")
        
        # Split data into training and validation sets
        train_size = int(0.8 * len(X))
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        # Initialize risk manager for simulated trading
        risk_manager = RiskManager(initial_balance=10000.0)
        
        # Training loop
        for episode in range(args.episodes):
            logger.info(f"Episode {episode+1}/{args.episodes}")
            total_reward = 0
            done = False
            
            # Start from a random position in the training data
            position = np.random.randint(0, len(X_train) - RL_CONFIG['episode_length'])
            
            # For DQN
            if args.model == 'dqn':
                for t in range(RL_CONFIG['episode_length']):
                    # Get current state
                    state = X_train[position + t].reshape(1, *state_size)
                    
                    # Choose action
                    action = model.act(state)
                    
                    # Calculate reward based on price movement and action
                    next_position = min(position + t + 1, len(X_train) - 1)
                    next_state = X_train[next_position].reshape(1, *state_size)
                    
                    price_change = y_train[next_position]
                    
                    # Reward logic: positive for correct prediction, negative for incorrect
                    if action == 0:  # Buy
                        reward = price_change * 100  # Scale for better learning
                    elif action == 1:  # Sell
                        reward = -price_change * 100
                    else:  # Hold
                        reward = -abs(price_change) * 10  # Smaller penalty for holding
                    
                    # Check if episode is done
                    done = (t == RL_CONFIG['episode_length'] - 1)
                    
                    # Remember experience
                    model.remember(state, action, reward, next_state, done)
                    
                    # Update total reward
                    total_reward += reward
                    
                    # Train the model
                    model.replay()
                
                logger.info(f"Episode {episode+1} - Total Reward: {total_reward}")
                
                # Save model periodically
                if (episode + 1) % 10 == 0:
                    model.save(f"{RL_CONFIG['model_save_path']}dqn_model_ep{episode+1}.h5")
            
            # For PPO
            else:
                for t in range(RL_CONFIG['episode_length']):
                    # Get current state
                    state = X_train[position + t].reshape(1, *state_size)
                    
                    # Choose action, get log probability and value
                    action, log_prob, value = model.act(state)
                    
                    # Calculate reward
                    next_position = min(position + t + 1, len(X_train) - 1)
                    price_change = y_train[next_position]
                    
                    # Reward logic: positive for correct prediction, negative for incorrect
                    if action == 0:  # Buy
                        reward = price_change * 100
                    elif action == 1:  # Sell
                        reward = -price_change * 100
                    else:  # Hold
                        reward = -abs(price_change) * 10
                    
                    # Check if episode is done
                    done = (t == RL_CONFIG['episode_length'] - 1)
                    
                    # Remember experience
                    model.remember(state, action, reward, value, log_prob, done)
                    
                    # Update total reward
                    total_reward += reward
                
                # Calculate final next state value for returns
                if position + RL_CONFIG['episode_length'] < len(X_train):
                    final_state = X_train[position + RL_CONFIG['episode_length']].reshape(1, *state_size)
                    _, _, final_value = model.act(final_state)
                else:
                    final_value = 0
                
                # Train PPO with collected experiences
                model.train(next_value=final_value)
                
                logger.info(f"Episode {episode+1} - Total Reward: {total_reward}")
                
                # Save model periodically
                if (episode + 1) % 10 == 0:
                    model.save(f"{RL_CONFIG['model_save_path']}ppo_model_ep{episode+1}_actor", 
                              f"{RL_CONFIG['model_save_path']}ppo_model_ep{episode+1}_critic")
        
        logger.info("Training completed")
    
    # Backtesting mode
    elif args.mode == 'backtest':
        logger.info("Starting backtest mode")
        
        # Initialize backtesting simulator
        simulator = BacktestSimulator(preprocessed_data, model, initial_balance=10000.0)
        
        # Run backtest
        results = simulator.run_backtest(sequence_length=sequence_length, state_size=state_size[1])
        
        # Log results
        logger.info("Backtest Results:")
        for key, value in results.items():
            logger.info(f"{key}: {value}")
    
    # Live trading mode
    elif args.mode == 'live':
        logger.info("Starting live trading mode")
        
        # Initialize exchange API and risk manager
        try:
            exchange_api = ExchangeAPI()
            risk_manager = RiskManager(initial_balance=10000.0)  # You may want to set this based on actual balance
            
            # Main trading loop
            while True:
                try:
                    # Fetch latest data
                    latest_data = data_collector.fetch_latest_data(limit=sequence_length+10)
                    
                    if latest_data is not None and len(latest_data) >= sequence_length:
                        # Preprocess latest data
                        preprocessed_latest_data = data_preprocessor.add_technical_indicators(latest_data)
                        normalized_data = data_preprocessor.normalize_data(preprocessed_latest_data)
                        
                        # Prepare state for model
                        latest_sequence = normalized_data.iloc[-sequence_length:].values.reshape(1, sequence_length, -1)
                        
                        # Get current price
                        current_price = latest_data.iloc[-1]['close']
                        
                        # Get model's action
                        if args.model == 'dqn':
                            action = model.act(latest_sequence, training=False)
                        else:
                            action = model.act(latest_sequence, training=False)
                        
                        # Convert action to trading signal
                        if action == 0:  # Buy
                            signal = 'buy'
                        elif action == 1:  # Sell
                            signal = 'sell'
                        else:  # Hold
                            signal = 'hold'
                            
                        logger.info(f"Signal: {signal}, Current Price: {current_price}")
                        
                        # Check current positions and risk
                        symbol = BOT_CONFIG['symbol']
                        
                        # Check stop loss and take profit for existing positions
                        if symbol in risk_manager.open_positions:
                            triggered, trade = risk_manager.check_stop_loss_take_profit(symbol, current_price)
                            if triggered:
                                # Execute the exit
                                side = 'sell' if trade['action'] == 'buy' else 'buy'
                                order = exchange_api.place_market_order(side, trade['amount'])
                                logger.info(f"Exit executed: {order}")
                            
                            # Update trailing stop if applicable
                            risk_manager.update_trailing_stop(symbol, current_price)
                        
                        # Execute new trades based on signal
                        if signal in ['buy', 'sell'] and symbol not in risk_manager.open_positions:
                            # Calculate position size
                            position_size = risk_manager.calculate_position_size(symbol, current_price)
                            
                            if position_size > 0:
                                # Execute the trade
                                order = exchange_api.place_market_order(signal, position_size)
                                if order:
                                    risk_manager.enter_position(symbol, current_price, position_size, signal)
                                    logger.info(f"Entry executed: {order}")
                        
                        elif signal == 'sell' and symbol in risk_manager.open_positions and risk_manager.open_positions[symbol]['action'] == 'buy':
                            # Exit long position
                            position = risk_manager.open_positions[symbol]
                            order = exchange_api.place_market_order('sell', position['amount'])
                            if order:
                                trade = risk_manager.exit_position(symbol, current_price, 'signal')
                                logger.info(f"Exit executed: {order}")
                        
                        elif signal == 'buy' and symbol in risk_manager.open_positions and risk_manager.open_positions[symbol]['action'] == 'sell':
                            # Exit short position
                            position = risk_manager.open_positions[symbol]
                            order = exchange_api.place_market_order('buy', position['amount'])
                            if order:
                                trade = risk_manager.exit_position(symbol, current_price, 'signal')
                                logger.info(f"Exit executed: {order}")
                    
                    # Sleep to avoid excessive API calls
                    time.sleep(60)  # Check every minute
                
                except Exception as e:
                    logger.error(f"Error in trading loop: {str(e)}")
                    time.sleep(60)
        
        except KeyboardInterrupt:
            logger.info("Live trading stopped by user")
            # Close all positions before shutting down
            for symbol in list(risk_manager.open_positions.keys()):
                exchange_api.close_all_positions(symbol)
                logger.info(f"Closed all positions for {symbol}")
        
        except Exception as e:
            logger.error(f"Error in live trading mode: {str(e)}")

if __name__ == "__main__":
    main()