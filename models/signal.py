from dataclasses import dataclass
from typing import Optional

@dataclass
class Signal:
    """Signal de trading généré par une stratégie"""
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0-1
    quantity: Optional[float] = None
    price: Optional[float] = None