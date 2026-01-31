"""Test script to evaluate the pair trading algorithm efficiency."""

from src.performance_simulator import PerformanceSimulator, SimulationConfig
from pathlib import Path
import time

def test_algorithm():
    """Run a test simulation and report performance metrics."""
    
    print('\n' + '='*80)
    print('🧪 Testing Pair Trading Algorithm on Tech Stocks')
    print('='*80)
    
    # Data directory (3 years of daily data: 2022-2024)
    data_dir = 'data/yahoo_tech_full'
    
    # Create configuration for stock pairs (adjusted for daily data)
    # With daily data, we count in days not hours, so divide by 24
    config = SimulationConfig(
        initial_capital=100_000,
        reference_symbol='AAPLUSDT',  # Apple as reference
        interval='1d',
        alpha1=0.15,  # Entry threshold  
        alpha2=0.08,  # Exit threshold
        formation_hours=30,  # 30 days formation (for daily data, this is 30 bars)
        trading_hours=10,    # 10 days trading
        step_hours=5,        # 5 day rolling window
        capital_per_side=8_000,   # 8% per position
        max_positions=8,
        fee_rate=0.001,  # 0.1% trading fee
    )
    
    print(f'\n📊 Configuration:')
    print(f'  Data: {data_dir}')
    print(f'  Stocks: 10 tech stocks (AAPL, MSFT, GOOGL, AMZN, META, TSLA, NVDA, AMD, NFLX, INTC)')
    print(f'  Period: 2022-2024 (752 trading days)')
    print(f'  Reference: {config.reference_symbol}')
    print(f'  Initial Capital: ${config.initial_capital:,.0f}')
    print(f'  Entry threshold (α1): {config.alpha1}')
    print(f'  Exit threshold (α2): {config.alpha2}')
    print(f'  Formation period: {config.formation_hours} days')
    print(f'  Trading period: {config.trading_hours} days')
    print(f'  Step size: {config.step_hours} days')
    print(f'  Expected cycles: ~{(752 - 40) // 5} trading cycles')
    print()
    
    # Run simulation with timing
    print('⏳ Running simulation...\n')
    start_time = time.time()
    
    simulator = PerformanceSimulator(config)
    results = simulator.run_simulation(data_dir)
    
    elapsed = time.time() - start_time
    
    # Generate report
    report_dir = 'reports/test_run'
    simulator.generate_report(results, report_dir)
    
    print(f'\n' + '='*80)
    print(f'✅ Simulation Complete!')
    print(f'='*80)
    print(f'⏱️  Execution time: {elapsed:.2f} seconds')
    print(f'📁 Reports saved to: {report_dir}/')
    print()
    
    # Show quick summary
    if not list(Path(report_dir).glob('report_*.txt')):
        print("⚠️  No report files found")
        return
    
    latest_report = max(Path(report_dir).glob('report_*.txt'), key=lambda p: p.stat().st_mtime)
    
    print('📊 Quick Summary:')
    print('-' * 80)
    with open(latest_report) as f:
        content = f.read()
        # Extract key metrics
        for line in content.split('\n'):
            if any(keyword in line for keyword in [
                'Total Return', 'Sharpe Ratio', 'Max Drawdown',
                'Win Rate', 'Total Trades', 'Profit Factor'
            ]):
                print(f'  {line}')
    print('-' * 80)

if __name__ == '__main__':
    test_algorithm()
