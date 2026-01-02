import requests
import os
import json
import logging

logger = logging.getLogger(__name__)

class TelegramBot:
    def __init__(self, token=None, chat_id=None):
        self.token = token or os.getenv("TELEGRAM_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        self.base_url = f"https://api.telegram.org/bot{self.token}/sendMessage"

    def send_signal(self, signal_data: dict):
        """
        Send a trading signal to Telegram.
        """
        if not self.token or not self.chat_id:
            logger.warning("Telegram credentials not found. Signal not sent.")
            return

        # Format Message
        try:
            message = self._format_message(signal_data)
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "Markdown"
            }
            
            response = requests.post(self.base_url, json=payload, timeout=5)
            if response.status_code == 200:
                logger.info(f"Signal sent for {signal_data.get('symbol')}")
            else:
                logger.error(f"Failed to send Telegram message: {response.text}")
                
        except Exception as e:
            logger.error(f"Error sending Telegram signal: {e}")

    def _format_message(self, data: dict) -> str:
        """
        Format signal dictionary into specific JSON string block as requested.
        """
        # Ensure strict JSON format is visible
        json_str = json.dumps(data, indent=2)
        
        emoji = "🟢" if data.get('signal') == 'BUY' else "🔴"
        
        message = f"""
{emoji} **NEW SIGNAL GENERATED**

```json
{json_str}
```
"""
        return message

if __name__ == "__main__":
    # Test
    bot = TelegramBot("TEST_TOKEN", "TEST_CHAT_ID")
    sample_signal = {
      "timestamp": "2023-10-27 10:00:00",
      "symbol": "BTC/USDT",
      "signal": "BUY",
      "confidence": 0.85,
      "suggested_entry": 30500,
      "suggested_sl": 30200,
      "suggested_tp": 31200,
      "strategy": "RSI_Mean_Reversion",
      "timeframe": "H1",
      "risk_reward": 2.5,
      "context": "RSI oversold + bullish divergence"
    }
    # bot.send_signal(sample_signal)
    print(bot._format_message(sample_signal))
