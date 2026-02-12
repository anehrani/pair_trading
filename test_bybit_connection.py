"""Test script for Bybit API connectivity.

Run this to verify your API credentials and test basic functionality.
"""

import os
import sys

from dotenv import load_dotenv
from loguru import logger

from src.bybit_client import BybitClient


def test_connection():
    """Test Bybit API connection."""
    logger.info("=== Testing Bybit API Connection ===")

    # Load environment variables
    load_dotenv()

    api_key = os.getenv("BYBIT_API_KEY")
    api_secret = os.getenv("BYBIT_API_SECRET")

    if not api_key or not api_secret:
        logger.error("BYBIT_API_KEY and BYBIT_API_SECRET must be set in .env file")
        logger.info("1. Copy .env.template to .env")
        logger.info("2. Get testnet API credentials from https://testnet.bybit.com/app/user/api-management")
        logger.info("3. Fill in your credentials in .env")
        return False

    try:
        # Initialize client
        client = BybitClient(api_key=api_key, api_secret=api_secret, testnet=True)

        # Test 1: Get account balance
        logger.info("Test 1: Getting account balance...")
        balance = client.get_account_balance()
        logger.success(f"✓ Balance: {balance}")

        # Test 2: Get latest price
        logger.info("Test 2: Getting latest BTC price...")
        price = client.get_latest_price("BTCUSDT")
        logger.success(f"✓ BTCUSDT price: ${price:,.2f}")

        # Test 3: Get historical klines
        logger.info("Test 3: Fetching historical klines...")
        df = client.get_klines("BTCUSDT", interval="5", limit=10)
        logger.success(f"✓ Fetched {len(df)} klines")
        logger.info(f"Latest candle: {df.iloc[-1]['timestamp']} - Close: ${df.iloc[-1]['close']:,.2f}")

        # Test 4: Get positions
        logger.info("Test 4: Getting open positions...")
        positions = client.get_positions()
        logger.success(f"✓ Open positions: {len(positions)}")
        if positions:
            for pos in positions:
                logger.info(f"  {pos.symbol}: {pos.side} {pos.size} @ ${pos.entry_price:,.2f}")

        logger.success("\\n=== All tests passed! ===")
        logger.info("You can now run the live trader with: .venv/bin/python src/live_trader.py")
        return True

    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
