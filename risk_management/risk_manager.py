import numpy as np
from datetime import datetime
from config.settings import RISK_CONFIG
import logging

class RiskManager:
    def __init__(self, initial_balance=10000.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.max_position_size = RISK_CONFIG['max_position_size']
        self.stop_loss_pct = RISK_CONFIG['stop_loss_percentage']
        self.take_profit_pct = RISK_CONFIG['take_profit_percentage']
        self.max_open_trades = RISK_CONFIG['max_open_trades']
        self.max_daily_loss_pct = RISK_CONFIG['max_daily_loss']
        
        # Trade tracking
        self.open_positions = {}  # symbol -> {entry_price, amount, stop_loss, take_profit}
        self.daily_trades = []  # List of [timestamp, profit_loss]
        self.trade_history = []  # All closed trades
        
        # Daily loss tracking
        self.day_start_balance = initial_balance
        self.current_day = datetime.now().date()
        
        # Logger
        self.logger = logging.getLogger('RiskManager')
    
    def _update_daily_metrics(self):
        """Update daily metrics and reset if day changed"""
        today = datetime.now().date()
        
        if today != self.current_day:
            self.day_start_balance = self.current_balance
            self.daily_trades = []
            self.current_day = today
            self.logger.info(f"New trading day started. Balance: {self.day_start_balance}")
    
    def calculate_position_size(self, symbol, entry_price):
        """Calculate the appropriate position size based on risk parameters"""
        self._update_daily_metrics()
        
        # Check if we're at the maximum number of open trades
        if len(self.open_positions) >= self.max_open_trades:
            self.logger.warning("Max number of open trades reached")
            return 0.0
        
        # Check if we've reached the maximum daily loss
        daily_loss_pct = (self.current_balance - self.day_start_balance) / self.day_start_balance
        if daily_loss_pct <= -self.max_daily_loss_pct:
            self.logger.warning(f"Maximum daily loss of {self.max_daily_loss_pct*100}% reached")
            return 0.0
        
        # Calculate position size as a percentage of current balance
        max_amount = self.current_balance * self.max_position_size / entry_price
        
        # Risk adjustment: reduce position size as daily losses increase
        if daily_loss_pct < 0:
            # Scale down linearly based on how close we are to max daily loss
            risk_factor = 1.0 - (abs(daily_loss_pct) / self.max_daily_loss_pct)
            max_amount *= max(0.1, risk_factor)  # Never go below 10% of normal size
        
        return max_amount
    
    def enter_position(self, symbol, entry_price, amount, action):
        """Record a new position"""
        if symbol in self.open_positions:
            self.logger.warning(f"Position already exists for {symbol}")
            return False
        
        # Calculate stop loss and take profit levels
        if action == "buy":  # Long position
            stop_loss = entry_price * (1 - self.stop_loss_pct)
            take_profit = entry_price * (1 + self.take_profit_pct)
        else:  # Short position
            stop_loss = entry_price * (1 + self.stop_loss_pct)
            take_profit = entry_price * (1 - self.take_profit_pct)
        
        # Record the position
        self.open_positions[symbol] = {
            'entry_price': entry_price,
            'amount': amount,
            'action': action,  # "buy" or "sell"
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_time': datetime.now()
        }
        
        # Update account balance
        cost = entry_price * amount
        self.current_balance -= cost
        
        self.logger.info(f"Entered {action} position for {symbol}: {amount} at {entry_price}")
        self.logger.info(f"Stop loss: {stop_loss}, Take profit: {take_profit}")
        
        return True
    
    def exit_position(self, symbol, exit_price, reason="manual"):
        """Close an existing position and record the result"""
        if symbol not in self.open_positions:
            self.logger.warning(f"No open position for {symbol} to exit")
            return False
        
        position = self.open_positions[symbol]
        entry_price = position['entry_price']
        amount = position['amount']
        action = position['action']
        
        # Calculate profit/loss
        if action == "buy":  # Long position
            profit_loss = (exit_price - entry_price) * amount
        else:  # Short position
            profit_loss = (entry_price - exit_price) * amount
        
        # Update account balance
        self.current_balance += (exit_price * amount) + profit_loss
        
        # Record the trade
        trade_record = {
            'symbol': symbol,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'amount': amount,
            'action': action,
            'profit_loss': profit_loss,
            'profit_loss_pct': profit_loss / (entry_price * amount),
            'entry_time': position['entry_time'],
            'exit_time': datetime.now(),
            'duration': (datetime.now() - position['entry_time']).total_seconds() // 60,  # minutes
            'reason': reason
        }
        
        self.trade_history.append(trade_record)
        self.daily_trades.append([datetime.now(), profit_loss])
        
        # Remove the position
        del self.open_positions[symbol]
        
        self.logger.info(f"Exited {action} position for {symbol} at {exit_price}")
        self.logger.info(f"Profit/Loss: {profit_loss} ({profit_loss / (entry_price * amount) * 100:.2f}%)")
        
        return trade_record
    
    def check_stop_loss_take_profit(self, symbol, current_price):
        """Check if stop loss or take profit has been triggered"""
        if symbol not in self.open_positions:
            return False, None
        
        position = self.open_positions[symbol]
        
        if position['action'] == "buy":  # Long position
            # Check stop loss
            if current_price <= position['stop_loss']:
                self.logger.info(f"Stop loss triggered for {symbol}")
                return True, self.exit_position(symbol, current_price, "stop_loss")
            
            # Check take profit
            if current_price >= position['take_profit']:
                self.logger.info(f"Take profit triggered for {symbol}")
                return True, self.exit_position(symbol, current_price, "take_profit")
        
        else:  # Short position
            # Check stop loss
            if current_price >= position['stop_loss']:
                self.logger.info(f"Stop loss triggered for {symbol}")
                return True, self.exit_position(symbol, current_price, "stop_loss")
            
            # Check take profit
            if current_price <= position['take_profit']:
                self.logger.info(f"Take profit triggered for {symbol}")
                return True, self.exit_position(symbol, current_price, "take_profit")
        
        return False, None
    
    def update_trailing_stop(self, symbol, current_price, trail_percent=0.015):
        """Update trailing stop loss if price moves favorably"""
        if symbol not in self.open_positions:
            return False
        
        position = self.open_positions[symbol]
        
        if position['action'] == "buy":  # Long position
            # Calculate potential new stop loss
            new_stop_loss = current_price * (1 - trail_percent)
            
            # Only move stop loss up, never down
            if new_stop_loss > position['stop_loss']:
                old_stop = position['stop_loss']
                position['stop_loss'] = new_stop_loss
                self.logger.info(f"Trailing stop updated for {symbol}: {old_stop} -> {new_stop_loss}")
                return True
        
        else:  # Short position
            # Calculate potential new stop loss
            new_stop_loss = current_price * (1 + trail_percent)
            
            # Only move stop loss down, never up
            if new_stop_loss < position['stop_loss']:
                old_stop = position['stop_loss']
                position['stop_loss'] = new_stop_loss
                self.logger.info(f"Trailing stop updated for {symbol}: {old_stop} -> {new_stop_loss}")
                return True
        
        return False
    
    def get_performance_metrics(self):
        """Calculate overall performance metrics"""
        if not self.trade_history:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'win_rate': 0.0,
                'total_profit_loss': 0.0,
                'profit_loss_percent': 0.0,
                'average_win': 0.0,
                'average_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'profit_factor': 0.0
            }
        
        # Extract profit/loss from all trades
        profits = [trade['profit_loss'] for trade in self.trade_history]
        
        # Calculate metrics
        total_trades = len(profits)
        winning_trades = sum(1 for p in profits if p > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_profit_loss = sum(profits)
        profit_loss_percent = total_profit_loss / self.initial_balance * 100
        
        # Separate winning and losing trades
        winning_profits = [p for p in profits if p > 0]
        losing_profits = [p for p in profits if p <= 0]
        
        average_win = np.mean(winning_profits) if winning_profits else 0
        average_loss = np.mean(losing_profits) if losing_profits else 0
        largest_win = max(profits) if profits else 0
        largest_loss = min(profits) if profits else 0
        
        # Profit factor
        total_gains = sum(p for p in profits if p > 0)
        total_losses = abs(sum(p for p in profits if p < 0))
        profit_factor = total_gains / total_losses if total_losses != 0 else float('inf')
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'total_profit_loss': total_profit_loss,
            'profit_loss_percent': profit_loss_percent,
            'average_win': average_win,
            'average_loss': average_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'profit_factor': profit_factor
        }