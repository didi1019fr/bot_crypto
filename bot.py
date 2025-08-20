import ccxt
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime
import json
import os
from typing import Dict, List, Optional

class BinanceTradingBot:
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        """
        Initialise le bot de trading Binance
        
        Args:
            api_key: Clé API Binance
            api_secret: Secret API Binance
            testnet: True pour utiliser le testnet, False pour le trading réel
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        
        # Configuration de l'exchange
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'sandbox': testnet,  # True pour testnet
            'rateLimit': 1200,
            'enableRateLimit': True,
        })
        
        # Paramètres de trading
        self.symbol = 'BTC/USDT'
        self.timeframe = '5m'
        self.trade_amount = 0.001  # Montant en BTC à trader
        self.stop_loss_pct = 0.02  # 2% stop loss
        self.take_profit_pct = 0.04  # 4% take profit
        
        # Paramètres technique
        self.ma_short = 10  # Moyenne mobile courte
        self.ma_long = 20   # Moyenne mobile longue
        self.rsi_period = 14
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        
        # État du bot
        self.position = None
        self.last_signal = None
        self.portfolio = {}
        
        # Configuration du logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('trading_bot.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def get_balance(self) -> Dict:
        """Récupère le solde du compte"""
        try:
            balance = self.exchange.fetch_balance()
            return balance
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération du solde: {e}")
            return {}
    
    def get_historical_data(self, limit: int = 100) -> pd.DataFrame:
        """
        Récupère les données historiques
        
        Args:
            limit: Nombre de bougies à récupérer
            
        Returns:
            DataFrame avec les données OHLCV
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(
                self.symbol, 
                self.timeframe, 
                limit=limit
            )
            
            df = pd.DataFrame(
                ohlcv, 
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des données: {e}")
            return pd.DataFrame()
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule les indicateurs techniques
        
        Args:
            df: DataFrame avec les données OHLCV
            
        Returns:
            DataFrame avec les indicateurs ajoutés
        """
        # Moyennes mobiles
        df['ma_short'] = df['close'].rolling(window=self.ma_short).mean()
        df['ma_long'] = df['close'].rolling(window=self.ma_long).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> str:
        """
        Génère les signaux de trading
        
        Args:
            df: DataFrame avec les indicateurs
            
        Returns:
            Signal: 'BUY', 'SELL', ou 'HOLD'
        """
        if len(df) < max(self.ma_long, self.rsi_period):
            return 'HOLD'
        
        latest = df.iloc[-1]
        previous = df.iloc[-2]
        
        buy_signals = 0
        sell_signals = 0
        
        # Signal 1: Croisement des moyennes mobiles
        if (latest['ma_short'] > latest['ma_long'] and 
            previous['ma_short'] <= previous['ma_long']):
            buy_signals += 1
        elif (latest['ma_short'] < latest['ma_long'] and 
              previous['ma_short'] >= previous['ma_long']):
            sell_signals += 1
        
        # Signal 2: RSI
        if latest['rsi'] < self.rsi_oversold:
            buy_signals += 1
        elif latest['rsi'] > self.rsi_overbought:
            sell_signals += 1
        
        # Signal 3: MACD
        if (latest['macd'] > latest['macd_signal'] and 
            previous['macd'] <= previous['macd_signal']):
            buy_signals += 1
        elif (latest['macd'] < latest['macd_signal'] and 
              previous['macd'] >= previous['macd_signal']):
            sell_signals += 1
        
        # Signal 4: Bollinger Bands
        if latest['close'] < latest['bb_lower']:
            buy_signals += 1
        elif latest['close'] > latest['bb_upper']:
            sell_signals += 1
        
        # Décision finale
        if buy_signals >= 2 and sell_signals == 0:
            return 'BUY'
        elif sell_signals >= 2 and buy_signals == 0:
            return 'SELL'
        else:
            return 'HOLD'
    
    def place_order(self, side: str, amount: float, price: Optional[float] = None) -> Dict:
        """
        Place un ordre
        
        Args:
            side: 'buy' ou 'sell'
            amount: Montant à trader
            price: Prix limite (None pour ordre au marché)
            
        Returns:
            Informations sur l'ordre
        """
        try:
            if price is None:
                # Ordre au marché
                order = self.exchange.create_market_order(
                    self.symbol, side, amount
                )
            else:
                # Ordre limite
                order = self.exchange.create_limit_order(
                    self.symbol, side, amount, price
                )
            
            self.logger.info(f"Ordre placé: {side} {amount} {self.symbol} à {price or 'prix marché'}")
            return order
        except Exception as e:
            self.logger.error(f"Erreur lors du placement de l'ordre: {e}")
            return {}
    
    def manage_position(self, current_price: float):
        """
        Gère la position ouverte (stop loss, take profit)
        
        Args:
            current_price: Prix actuel
        """
        if not self.position:
            return
        
        entry_price = self.position['entry_price']
        side = self.position['side']
        amount = self.position['amount']
        
        if side == 'buy':
            # Position longue
            stop_loss_price = entry_price * (1 - self.stop_loss_pct)
            take_profit_price = entry_price * (1 + self.take_profit_pct)
            
            if current_price <= stop_loss_price:
                self.logger.info(f"Stop loss déclenché à {current_price}")
                self.place_order('sell', amount)
                self.position = None
            elif current_price >= take_profit_price:
                self.logger.info(f"Take profit déclenché à {current_price}")
                self.place_order('sell', amount)
                self.position = None
        
        elif side == 'sell':
            # Position courte
            stop_loss_price = entry_price * (1 + self.stop_loss_pct)
            take_profit_price = entry_price * (1 - self.take_profit_pct)
            
            if current_price >= stop_loss_price:
                self.logger.info(f"Stop loss déclenché à {current_price}")
                self.place_order('buy', amount)
                self.position = None
            elif current_price <= take_profit_price:
                self.logger.info(f"Take profit déclenché à {current_price}")
                self.place_order('buy', amount)
                self.position = None
    
    def run_strategy(self):
        """Exécute une itération de la stratégie"""
        try:
            # Récupérer les données
            df = self.get_historical_data()
            if df.empty:
                return
            
            # Calculer les indicateurs
            df = self.calculate_indicators(df)
            
            # Prix actuel
            current_price = df['close'].iloc[-1]
            
            # Gérer la position existante
            self.manage_position(current_price)
            
            # Générer un nouveau signal si pas de position
            if not self.position:
                signal = self.generate_signals(df)
                
                if signal == 'BUY' and self.last_signal != 'BUY':
                    self.logger.info(f"Signal d'achat généré à {current_price}")
                    order = self.place_order('buy', self.trade_amount)
                    if order:
                        self.position = {
                            'side': 'buy',
                            'amount': self.trade_amount,
                            'entry_price': current_price,
                            'timestamp': datetime.now()
                        }
                    self.last_signal = 'BUY'
                
                elif signal == 'SELL' and self.last_signal != 'SELL':
                    self.logger.info(f"Signal de vente généré à {current_price}")
                    order = self.place_order('sell', self.trade_amount)
                    if order:
                        self.position = {
                            'side': 'sell',
                            'amount': self.trade_amount,
                            'entry_price': current_price,
                            'timestamp': datetime.now()
                        }
                    self.last_signal = 'SELL'
                
                elif signal == 'HOLD':
                    self.last_signal = 'HOLD'
            
            # Log des informations importantes
            self.logger.info(f"Prix: {current_price:.2f}, Signal: {self.last_signal}, Position: {bool(self.position)}")
            
        except Exception as e:
            self.logger.error(f"Erreur dans la stratégie: {e}")
    
    def run(self, interval: int = 300):
        """
        Lance le bot en continu
        
        Args:
            interval: Intervalle entre les vérifications en secondes
        """
        self.logger.info("Démarrage du bot de trading...")
        
        # Vérifier la connexion
        try:
            balance = self.get_balance()
            self.logger.info(f"Connexion réussie. Solde: {balance.get('USDT', {}).get('free', 0)} USDT")
        except Exception as e:
            self.logger.error(f"Erreur de connexion: {e}")
            return
        
        while True:
            try:
                self.run_strategy()
                time.sleep(interval)
            except KeyboardInterrupt:
                self.logger.info("Arrêt du bot...")
                break
            except Exception as e:
                self.logger.error(f"Erreur inattendue: {e}")
                time.sleep(30)  # Attendre 30s en cas d'erreur

# Configuration et lancement
if __name__ == "__main__":
    # ATTENTION: Remplacez par vos vraies clés API
    API_KEY = "your_binance_api_key"
    API_SECRET = "your_binance_api_secret"
    
    # Créer et lancer le bot
    bot = BinanceTradingBot(
        api_key=API_KEY,
        api_secret=API_SECRET,
        testnet=True  # Utilisez False pour le trading réel
    )
    
    # Configuration personnalisée (optionnel)
    bot.symbol = 'BTC/USDT'
    bot.trade_amount = 0.001
    bot.stop_loss_pct = 0.02
    bot.take_profit_pct = 0.04
    
    # Lancer le bot (vérifie toutes les 5 minutes)
    bot.run(interval=300)
    