import ccxt
import time
from config.settings import BOT_CONFIG
import logging

class ExchangeAPI:
    def __init__(self):
        self.exchange_id = BOT_CONFIG["exchange"]
        self.symbol = BOT_CONFIG["symbol"]
        self.timeframe = BOT_CONFIG["timeframe"]
        
        # Initialize exchange
        self.exchange = getattr(ccxt, self.exchange_id)({
            'apiKey': BOT_CONFIG['api_key'],
            'secret': BOT_CONFIG['api_secret'],
            'enableRateLimit': True,
        })
        
        # Logger
        self.logger = logging.getLogger('ExchangeAPI')
        
        # Test connection
        try:
            self.exchange.load_markets()
            self.logger.info(f"Connected to {self.exchange_id}")
        except Exception as e:
            self.logger.error(f"Failed to connect to exchange: {e}")
            raise
    
    def get_balance(self):
        """Get account balance"""
        try:
            balance = self.exchange.fetch_balance()
            return balance
        except Exception as e:
            self.logger.error(f"Error fetching balance: {e}")
            return None
    
    def get_ticker(self, symbol=None):
        """Get current ticker information"""
        if symbol is None:
            symbol = self.symbol
            
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker
        except Exception as e:
            self.logger.error(f"Error fetching ticker for {symbol}: {e}")
            return None
    
    def place_market_order(self, side, amount, symbol=None):
        """Place a market order"""
        if symbol is None:
            symbol = self.symbol
            
        try:
            order = self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=side.lower(),  # buy or sell
                amount=amount
            )
            
            self.logger.info(f"Placed {side} market order for {amount} {symbol}")
            return order
        except Exception as e:
            self.logger.error(f"Error placing market order: {e}")
            return None
    
    def place_limit_order(self, side, amount, price, symbol=None):
        """Place a limit order"""
        if symbol is None:
            symbol = self.symbol
            
        try:
            order = self.exchange.create_order(
                symbol=symbol,
                type='limit',
                side=side.lower(),  # buy or sell
                amount=amount,
                price=price
            )
            
            self.logger.info(f"Placed {side} limit order for {amount} {symbol} at {price}")
            return order
        except Exception as e:
            self.logger.error(f"Error placing limit order: {e}")
            return None
    
    def cancel_order(self, order_id, symbol=None):
        """Cancel an existing order"""
        if symbol is None:
            symbol = self.symbol
            
        try:
            result = self.exchange.cancel_order(order_id, symbol)
            self.logger.info(f"Cancelled order {order_id} for {symbol}")
            return result
        except Exception as e:
            self.logger.error(f"Error cancelling order: {e}")
            return None
    
    def get_order_status(self, order_id, symbol=None):
        """Get the status of an order"""
        if symbol is None:
            symbol = self.symbol
            
        try:
            order = self.exchange.fetch_order(order_id, symbol)
            return order
        except Exception as e:
            self.logger.error(f"Error fetching order status: {e}")
            return None
    
    def get_open_orders(self, symbol=None):
        """Get all open orders"""
        if symbol is None:
            symbol = self.symbol
            
        try:
            orders = self.exchange.fetch_open_orders(symbol)
            return orders
        except Exception as e:
            self.logger.error(f"Error fetching open orders: {e}")
            return []
    
    def get_closed_orders(self, symbol=None, since=None, limit=None):
        """Get closed orders"""
        if symbol is None:
            symbol = self.symbol
            
        try:
            orders = self.exchange.fetch_closed_orders(symbol, since, limit)
            return orders
        except Exception as e:
            self.logger.error(f"Error fetching closed orders: {e}")
            return []
    
    def close_all_positions(self, symbol=None):
        """Close all open positions (useful for emergency shutdown)"""
        if symbol is None:
            symbol = self.symbol
            
        try:
            # Get balance of the quote and base currency
            balance = self.get_balance()
            
            # Extract symbol components (e.g. BTC/USDT -> BTC and USDT)
            base, quote = symbol.split('/')
            
            base_balance = balance['free'].get(base, 0)
            
            if base_balance > 0:
                # Convert to precision supported by the exchange
                market = self.exchange.market(symbol)
                amount = self.exchange.amount_to_precision(symbol, base_balance)
                
                # Place a market sell order to close the position
                order = self.place_market_order('sell', float(amount), symbol)
                self.logger.info(f"Closed all positions for {symbol}")
                return order
            
            return None
        except Exception as e:
            self.logger.error(f"Error closing positions: {e}")
            return None