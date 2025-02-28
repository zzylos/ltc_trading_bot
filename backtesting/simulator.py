import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from config.settings import BOT_CONFIG, DATA_CONFIG, RL_CONFIG
from risk_management.risk_manager import RiskManager
import logging

class BacktestSimulator:
    def __init__(self, data, model, initial_balance=10000.0):
        self.data = data
        self.model = model
        self.initial_balance = initial_balance
        
        # Risk management
        self.risk_manager = RiskManager(initial_balance)
        
        # Tracking variables
        self.current_position = None  # None, 'long', or 'short'
        self.entry_price = 0.0
        self.position_size = 0.0
        self.cash_balance = initial_balance
        self.equity = initial_balance
        self.equity_curve = []
        
        # Performance metrics
        self.trades = []
        self.daily_returns = []
        
        # Logger
        self.logger = logging.getLogger('BacktestSimulator')
        
        # Create directory for backtest results
        os.makedirs('backtest_results', exist_ok=True)
    
    def run_backtest(self, sequence_length=60, state_size=None):
        """Run the backtest on historical data"""
        self.logger.info("Starting backtest...")
        
        for i in range(sequence_length, len(self.data)):
            # Prepare state for model input
            if state_size:
                state = self.data[i-sequence_length:i].values.reshape(1, sequence_length, state_size)
            else:
                # If state_size is not provided, infer from data
                state = self.data[i-sequence_length:i].values.reshape(1, sequence_length, -1)
            
            # Get current price
            current_price = self.data.iloc[i]['close']
            timestamp = self.data.index[i]
            
            # Get model's action
            action = self.model.act(state, training=False)
            
            # Convert action to trading signal
            if action == 0:  # Buy
                signal = 'buy'
            elif action == 1:  # Sell
                signal = 'sell'
            else:  # Hold
                signal = 'hold'
                
            # Execute trading decisions
            self._execute_trade(signal, current_price, timestamp, i)
            
            # Update equity curve
            self._update_equity(current_price, timestamp)
            
        self.logger.info("Backtest completed")
        results = self._calculate_performance_metrics()
        self._generate_report(results)
        
        return results
    
    def _execute_trade(self, signal, current_price, timestamp, index):
        """Execute trading decisions based on signals"""
        # Check stop loss and take profit if we have a position
        if self.current_position:
            triggered, trade = self.risk_manager.check_stop_loss_take_profit(BOT_CONFIG['symbol'], current_price)
            if triggered:
                self.trades.append(trade)
                self.current_position = None
                
                # Log the trade
                self.logger.info(f"Exit at {timestamp}: Price={current_price}, Reason={trade['reason']}")
        
        # Process the signal if we don't have a position or if it's an exit signal
        if signal in ['buy', 'sell']:
            if self.current_position is None:  # Entry
                # Calculate position size
                position_size = self.risk_manager.calculate_position_size(BOT_CONFIG['symbol'], current_price)
                
                if position_size > 0:
                    self.risk_manager.enter_position(BOT_CONFIG['symbol'], current_price, position_size, signal)
                    self.current_position = 'long' if signal == 'buy' else 'short'
                    
                    # Log the trade
                    self.logger.info(f"Entry at {timestamp}: {signal.upper()}, Price={current_price}, Size={position_size}")
                    
            elif (self.current_position == 'long' and signal == 'sell') or \
                 (self.current_position == 'short' and signal == 'buy'):  # Exit
                
                trade = self.risk_manager.exit_position(BOT_CONFIG['symbol'], current_price, "signal")
                self.trades.append(trade)
                self.current_position = None
                
                # Log the trade
                self.logger.info(f"Exit at {timestamp}: Price={current_price}, Reason=signal")
    
    def _update_equity(self, current_price, timestamp):
        """Update equity based on current position and price"""
        # If we have a position, calculate unrealized profit/loss
        if self.current_position:
            position = self.risk_manager.open_positions.get(BOT_CONFIG['symbol'])
            if position:
                entry_price = position['entry_price']
                amount = position['amount']
                
                if position['action'] == 'buy':  # Long position
                    unrealized_pnl = (current_price - entry_price) * amount
                else:  # Short position
                    unrealized_pnl = (entry_price - current_price) * amount
                    
                self.equity = self.risk_manager.current_balance + (current_price * amount) + unrealized_pnl
            else:
                self.equity = self.risk_manager.current_balance
        else:
            self.equity = self.risk_manager.current_balance
        
        # Record equity for the equity curve
        self.equity_curve.append((timestamp, self.equity))
        
        # Calculate daily return if this is a new day
        if len(self.equity_curve) > 1:
            prev_day = self.equity_curve[-2][0].date()
            current_day = timestamp.date()
            
            if current_day > prev_day:
                prev_equity = self.equity_curve[-2][1]
                daily_return = (self.equity - prev_equity) / prev_equity
                self.daily_returns.append((current_day, daily_return))
    
    def _calculate_performance_metrics(self):
        """Calculate trading performance metrics"""
        # Basic metrics
        final_equity = self.equity
        total_return = (final_equity - self.initial_balance) / self.initial_balance
        
        # Convert equity curve to DataFrame for easier analysis
        equity_df = pd.DataFrame(self.equity_curve, columns=['timestamp', 'equity'])
        equity_df.set_index('timestamp', inplace=True)
        equity_df['returns'] = equity_df['equity'].pct_change()
        
        # Daily metrics
        daily_returns_df = pd.DataFrame(self.daily_returns, columns=['date', 'return'])
        daily_returns_df.set_index('date', inplace=True)
        
        # Calculate key metrics
        trading_days = len(daily_returns_df)
        winning_days = len(daily_returns_df[daily_returns_df['return'] > 0])
        
        if trading_days > 0:
            win_rate = winning_days / trading_days
        else:
            win_rate = 0
            
        # Sharpe Ratio (assuming risk-free rate of 0 for simplicity)
        if len(daily_returns_df) > 1:
            sharpe_ratio = np.mean(daily_returns_df['return']) / np.std(daily_returns_df['return']) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Maximum Drawdown
        equity_df['equity_peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['equity_peak']) / equity_df['equity_peak']
        max_drawdown = equity_df['drawdown'].min()
        
        # Compile all metrics
        risk_metrics = self.risk_manager.get_performance_metrics()
        
        return {
            'initial_balance': self.initial_balance,
            'final_equity': final_equity,
            'total_return': total_return,
            'total_return_percent': total_return * 100,
            'trading_days': trading_days,
            'winning_days': winning_days,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_percent': max_drawdown * 100,
            'total_trades': risk_metrics['total_trades'],
            'winning_trades': risk_metrics['winning_trades'],
            'trade_win_rate': risk_metrics['win_rate'],
            'average_win': risk_metrics['average_win'],
            'average_loss': risk_metrics['average_loss'],
            'profit_factor': risk_metrics['profit_factor']
        }
    
    def _generate_report(self, results):
        """Generate a backtest report with visualizations"""
        # Create a DataFrame for plotting the equity curve
        equity_df = pd.DataFrame(self.equity_curve, columns=['timestamp', 'equity'])
        equity_df.set_index('timestamp', inplace=True)
        
        # Plot equity curve
        plt.figure(figsize=(12, 6))
        plt.plot(equity_df.index, equity_df['equity'])
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Equity ($)')
        plt.grid(True)
        plt.savefig('backtest_results/equity_curve.png')
        
        # Plot drawdown
        equity_df['equity_peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['equity_peak']) / equity_df['equity_peak'] * 100
        
        plt.figure(figsize=(12, 6))
        plt.plot(equity_df.index, equity_df['drawdown'])
        plt.fill_between(equity_df.index, equity_df['drawdown'], 0, alpha=0.3, color='red')
        plt.title('Drawdown Percentage')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True)
        plt.savefig('backtest_results/drawdown.png')
        
        # Plot monthly returns
        if len(equity_df) > 30:
            equity_df['monthly_return'] = equity_df['equity'].resample('M').ffill().pct_change()
            monthly_returns = equity_df['monthly_return'].dropna()
            
            plt.figure(figsize=(12, 6))
            monthly_returns.plot(kind='bar')
            plt.title('Monthly Returns')
            plt.xlabel('Month')
            plt.ylabel('Return (%)')
            plt.grid(True)
            plt.savefig('backtest_results/monthly_returns.png')
        
        # Save trades to CSV
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_df.to_csv('backtest_results/trades.csv', index=False)
        
        # Save results to text file
        with open('backtest_results/performance_metrics.txt', 'w') as f:
            f.write("Performance Metrics\n")
            f.write("===================\n\n")
            f.write(f"Initial Balance: ${results['initial_balance']:.2f}\n")
            f.write(f"Final Equity: ${results['final_equity']:.2f}\n")
            f.write(f"Total Return: {results['total_return_percent']:.2f}%\n")
            f.write(f"Trading Days: {results['trading_days']}\n")
            f.write(f"Win Rate (Days): {results['win_rate']*100:.2f}%\n")
            f.write(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}\n")
            f.write(f"Max Drawdown: {results['max_drawdown_percent']:.2f}%\n")
            f.write(f"Total Trades: {results['total_trades']}\n")
            f.write(f"Winning Trades: {results['winning_trades']}\n")
            f.write(f"Trade Win Rate: {results['trade_win_rate']*100:.2f}%\n")
            f.write(f"Average Win: ${results['average_win']:.2f}\n")
            f.write(f"Average Loss: ${results['average_loss']:.2f}\n")
            f.write(f"Profit Factor: {results['profit_factor']:.2f}\n")
        
        self.logger.info(f"Backtest report saved to backtest_results/ directory")