"""Test script for OKX API connectivity.

Run this to verify your API credentials and test basic functionality.
"""

import os
import sys

from dotenv import load_dotenv
from loguru import logger

from src.okx_client import OKXClient


def test_connection():
    """Test OKX API connection."""
    logger.info("=== Testing OKX API Connection ===")

    # Load environment variables
    load_dotenv()

    api_key = os.getenv("OKX_API_KEY")
    api_secret = os.getenv("OKX_API_SECRET")
    passphrase = os.getenv("OKX_PASSPHRASE")

    if not api_key or not api_secret or not passphrase:
        logger.error("OKX_API_KEY, OKX_API_SECRET, and OKX_PASSPHRASE must be set in .env file")
        return False

    try:
        # Initialize client
        client = OKXClient(api_key=api_key, api_secret=api_secret, passphrase=passphrase, demo=True)

        # Test 1: Get account balance
        logger.info("Test 1: Getting account balance...")
        balance = client.get_account_balance()
        logger.success(f"✓ Balance: {balance}")

        # Test 2: Get latest price
        logger.info("Test 2: Getting latest BTC price...")
        price = client.get_latest_price("BTC-USDT")  # Use Spot for simple test
        logger.success(f"✓ BTC-USDT price: ${price:,.2f}")

        # Test 3: Get historical klines
        logger.info("Test 3: Fetching historical klines...")
        df = client.get_klines("BTC-USDT", interval="5m", limit=10)
        logger.success(f"✓ Fetched {len(df)} klines")
        if not df.empty:
            logger.info(f"Latest candle: {df.iloc[-1]['timestamp']} - Close: ${df.iloc[-1]['close']:,.2f}")

        # Test 4: Get positions
        logger.info("Test 4: Getting open positions...")
        positions = client.get_positions()
        logger.success(f"✓ Open positions: {len(positions)}")
        if positions:
            for pos in positions:
                logger.info(f"  {pos.symbol}: {pos.side} {pos.size} @ ${pos.entry_price:,.2f}")

        logger.success("\n=== All tests passed! ===")
        logger.info("You can now run the live trader with: .venv/bin/python src/live_trader.py")
        return True

    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
