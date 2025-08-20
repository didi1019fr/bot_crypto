from dataclasses import dataclass
from datetime import datetime

@dataclass
class Trade:
    """Représente une transaction"""
    symbol: str
    side: str  # 'BUY' ou 'SELL'
    quantity: float
    price: float
    timestamp: datetime
    fees: float = 0.0