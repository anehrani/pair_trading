"""Check if proxy or IP restrictions are causing issues."""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("Checking Network Configuration")
print("=" * 60)
print()

# Check proxy settings
print("1. Proxy Environment Variables:")
proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY']
found_proxy = False
for var in proxy_vars:
    val = os.getenv(var)
    if val:
        print(f"   {var}: {val}")
        found_proxy = True
if not found_proxy:
    print("   No proxy variables set")
print()

# Check public IP
print("2. Your Public IP Address:")
try:
    ip = requests.get('https://api.ipify.org', timeout=5).text
    print(f"   {ip}")
except Exception as e:
    print(f"   Failed to get IP: {e}")
print()

# Test Bybit connectivity
print("3. Testing Bybit Testnet Connectivity:")
try:
    resp = requests.get('https://api-testnet.bybit.com/v5/market/time', timeout=5)
    if resp.status_code == 200:
        print(f"   ✓ Can reach Bybit testnet")
        print(f"   Server time: {resp.json()}")
    else:
        print(f"   ✗ Got status code: {resp.status_code}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
print()

print("=" * 60)
print("NEXT STEPS:")
print("=" * 60)
print()
print("If you're using a proxy or VPN:")
print("  1. Go to Bybit API settings")
print("  2. Check 'IP Restrictions' section")
print("  3. Either:")
print("     a) Leave it EMPTY (allow all IPs), OR")
print("     b) Add your current IP:", end=" ")
try:
    print(requests.get('https://api.ipify.org', timeout=5).text)
except:
    print("(see above)")
print()
print("If using SSH key authentication:")
print("  - Make sure you pasted the FULL public key")
print("  - Including 'ssh-ed25519' at the start")
print("  - And email at the end")
print()
