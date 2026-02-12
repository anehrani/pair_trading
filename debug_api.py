"""Test API key permissions and try simpler endpoints."""

import os
from dotenv import load_dotenv
from pybit.unified_trading import HTTP

load_dotenv()

api_key = os.getenv("BYBIT_API_KEY")
api_secret = os.getenv("BYBIT_API_SECRET")

session = HTTP(
    testnet=True,
    api_key=api_key,
    api_secret=api_secret,
)

print("=" * 60)
print("Testing Bybit API Key Permissions")
print("=" * 60)
print()

# Test 1: Get API key info (should work if key is valid)
print("1. Testing API key info...")
try:
    result = session.get_api_key_information()
    print(f"   ✓ Success! API Key is valid")
    print(f"   Response: {result}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
print()

# Test 2: Get positions (simpler than wallet balance)
print("2. Testing get positions...")
try:
    result = session.get_positions(category="linear", settleCoin="USDT")
    print(f"   ✓ Success!")
    print(f"   Response: {result}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
print()

# Test 3: Get tickers (public endpoint, no auth)
print("3. Testing get tickers (public, no auth)...")
try:
    result = session.get_tickers(category="linear", symbol="BTCUSDT")
    print(f"   ✓ Success!")
    print(f"   BTC Price: ${float(result['result']['list'][0]['lastPrice']):,.2f}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
print()

# Test 4: Get klines (public endpoint)
print("4. Testing get klines (public, no auth)...")
try:
    result = session.get_kline(category="linear", symbol="BTCUSDT", interval="5", limit=1)
    print(f"   ✓ Success! Fetched kline data")
except Exception as e:
    print(f"   ✗ Failed: {e}")
print()

print("=" * 60)
print("DIAGNOSIS:")
print("=" * 60)
print()
print("If tests 3-4 passed but 1-2 failed:")
print("  → API key doesn't have required permissions")
print("  → Go to https://testnet.bybit.com/app/user/api-management")
print("  → Edit your API key and enable:")
print("     - 'Read-Write' for Derivatives (Contract Trading)")
print("     - 'Read-Write' for Wallet")
print()
print("If all tests failed:")
print("  → API key/secret might be incorrect")
print("  → Make sure you're using TESTNET credentials")
print()
