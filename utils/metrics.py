import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

class PerformanceAnalyzer:
    def __init__(self, trades=None, equity_curve=None):
        self.trades = trades  # List of trade dictionaries
        self.equity_curve = equity_curve  # List of (timestamp, equity) tuples
        
        # Create results directory if it doesn't exist
        if not os.path.exists('analysis_results'):
            os.makedirs('analysis_results')
    
    def calculate_metrics(self):
        """Calculate comprehensive performance metrics"""
        if not self.trades:
            return None
        
        # Convert trades to DataFrame
        trades_df = pd.DataFrame(self.trades)
        
        # Basic trade metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['profit_loss'] > 0])
        losing_trades = len(trades_df[trades_df['profit_loss'] <= 0])
        
        # Profit metrics
        total_profit = trades_df[trades_df['profit_loss'] > 0]['profit_loss'].sum()
        total_loss = trades_df[trades_df['profit_loss'] <= 0]['profit_loss'].sum()
        net_profit = total_profit + total_loss
        
        # Ratios
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
        
        # Average metrics
        avg_profit = total_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = total_loss / losing_trades if losing_trades > 0 else 0
        avg_trade = net_profit / total_trades if total_trades > 0 else 0
        
        # Risk metrics
        if self.equity_curve:
            equity_df = pd.DataFrame(self.equity_curve, columns=['timestamp', 'equity'])
            equity_df.set_index('timestamp', inplace=True)
            equity_df['return'] = equity_df['equity'].pct_change()
            
            # Calculate drawdown
            equity_df['peak'] = equity_df['equity'].cummax()
            equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak']
            max_drawdown = equity_df['drawdown'].min()
            
            # Calculate Sharpe ratio (assuming risk-free rate of 0)
            sharpe_ratio = np.sqrt(252) * (equity_df['return'].mean() / equity_df['return'].std())
            
            # Calculate Calmar ratio
            annual_return = (equity_df['equity'].iloc[-1] / equity_df['equity'].iloc[0]) ** (252 / len(equity_df)) - 1
            calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else float('inf')
            
        else:
            max_drawdown = None
            sharpe_ratio = None
            calmar_ratio = None
        
        # Create metrics dictionary
        metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'net_profit': net_profit,
            'profit_factor': profit_factor,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'avg_trade': avg_trade,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': calmar_ratio
        }
        
        return metrics
    
    def generate_report(self, metrics=None, save=True):
        """Generate a comprehensive performance report"""
        if metrics is None:
            metrics = self.calculate_metrics()
            
        if metrics is None:
            print("No trades or metrics available for analysis.")
            return
        
        # Create report
        report = "Trading Performance Report\n"
        report += "=========================\n\n"
        
        report += "Trade Statistics:\n"
        report += f"  Total Trades: {metrics['total_trades']}\n"
        report += f"  Winning Trades: {metrics['winning_trades']}\n"
        report += f"  Losing Trades: {metrics['losing_trades']}\n"
        report += f"  Win Rate: {metrics['win_rate']*100:.2f}%\n\n"
        
        report += "Profit Statistics:\n"
        report += f"  Total Profit: ${metrics['total_profit']:.2f}\n"
        report += f"  Total Loss: ${metrics['total_loss']:.2f}\n"
        report += f"  Net Profit: ${metrics['net_profit']:.2f}\n"
        report += f"  Profit Factor: {metrics['profit_factor']:.2f}\n\n"
        
        report += "Trade Averages:\n"
        report += f"  Average Profit: ${metrics['avg_profit']:.2f}\n"
        report += f"  Average Loss: ${metrics['avg_loss']:.2f}\n"
        report += f"  Average Trade: ${metrics['avg_trade']:.2f}\n\n"
        
        report += "Risk Metrics:\n"
        if metrics['max_drawdown'] is not None:
            report += f"  Maximum Drawdown: {metrics['max_drawdown']*100:.2f}%\n"
        if metrics['sharpe_ratio'] is not None:
            report += f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
        if metrics['calmar_ratio'] is not None:
            report += f"  Calmar Ratio: {metrics['calmar_ratio']:.2f}\n"
            
        # Print report
        print(report)
        
        # Save report
        if save and self.equity_curve:
            # Save report text
            with open('analysis_results/performance_report.txt', 'w') as f:
                f.write(report)
            
            # Generate and save plots
            self._generate_plots()
            
            print("Report and charts saved to analysis_results/ directory")
        
        return report
    
    def _generate_plots(self):
        """Generate and save performance charts"""
        if not self.equity_curve:
            return
            
        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(self.equity_curve, columns=['timestamp', 'equity'])
        equity_df.set_index('timestamp', inplace=True)
        
        # Equity curve plot
        plt.figure(figsize=(12, 6))
        plt.plot(equity_df.index, equity_df['equity'])
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Equity ($)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('analysis_results/equity_curve.png')
        
        # Drawdown plot
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak'] * 100
        
        plt.figure(figsize=(12, 6))
        plt.plot(equity_df.index, equity_df['drawdown'])
        plt.fill_between(equity_df.index, equity_df['drawdown'], 0, alpha=0.3, color='red')
        plt.title('Drawdown Percentage')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('analysis_results/drawdown.png')
        
        # Return distribution
        if len(equity_df) > 1:
            equity_df['return'] = equity_df['equity'].pct_change().dropna()
            
            plt.figure(figsize=(12, 6))
            plt.hist(equity_df['return'], bins=50, alpha=0.75)
            plt.axvline(x=0, color='r', linestyle='dashed', alpha=0.75)
            plt.title('Return Distribution')
            plt.xlabel('Daily Return')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('analysis_results/return_distribution.png')
        
        # Trade analysis if trades are available
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            
            # Profit/loss by trade
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(trades_df)), trades_df['profit_loss'])
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            plt.title('Profit/Loss by Trade')
            plt.xlabel('Trade Number')
            plt.ylabel('Profit/Loss ($)')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('analysis_results/trade_pnl.png')
            
            # Cumulative profit/loss
            plt.figure(figsize=(12, 6))
            plt.plot(np.cumsum(trades_df['profit_loss']))
            plt.title('Cumulative Profit/Loss')
            plt.xlabel('Trade Number')
            plt.ylabel('Cumulative P/L ($)')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('analysis_results/cumulative_pnl.png')