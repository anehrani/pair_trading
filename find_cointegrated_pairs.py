"""Find cointegrated pairs in the complete_data directory and generate a report."""

from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from src.stats_tests import cointegration_with_reference

def load_data(data_dir: str) -> dict[str, pd.Series]:
    """Load all price data from the directory."""
    data_path = Path(data_dir)
    all_series = {}
    
    for csv_file in sorted(data_path.glob("*.csv")):
        symbol = csv_file.stem.replace("_1d", "")
        try:
            # Read CSV with timestamp column
            df = pd.read_csv(csv_file)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                df = df.set_index('timestamp')
            else:
                # Try to use first column as index
                df = df.set_index(df.columns[0])
                df.index = pd.to_datetime(df.index, utc=True)
            
            if 'close' in df.columns and len(df) > 0:
                # Use close price
                close_series = pd.to_numeric(df['close'], errors='coerce')
                close_series = close_series.dropna()
                all_series[symbol] = close_series
                print(f"✓ Loaded {symbol}: {len(close_series)} bars")
            else:
                print(f"✗ Skipped {symbol}: no close column or empty data")
        except Exception as e:
            print(f"✗ Error loading {symbol}: {e}")
    
    return all_series

def find_all_cointegrated_pairs(
    data: dict[str, pd.Series],
    eg_alpha: float = 0.10,
    adf_alpha: float = 0.10,
    kss_critical: float = -1.92,
    use_intercept: bool = False,
    min_observations: int = 200
) -> list[dict]:
    """Find all cointegrated pairs in the dataset."""
    
    results = []
    symbols = list(data.keys())
    total_pairs = len(symbols) * (len(symbols) - 1) // 2
    
    print(f"\n🔍 Searching {total_pairs} possible pairs...")
    print(f"   Criteria: EG p-value < {eg_alpha}, ADF p-value < {adf_alpha}, KSS < {kss_critical}")
    print()
    
    checked = 0
    for i, ref_symbol in enumerate(symbols):
        for asset_symbol in symbols[i+1:]:
            checked += 1
            if checked % 50 == 0:
                print(f"   Checked {checked}/{total_pairs} pairs...")
            
            ref_series = data[ref_symbol]
            asset_series = data[asset_symbol]
            
            # Align series
            ref_aligned, asset_aligned = ref_series.align(asset_series, join='inner')
            
            if len(ref_aligned) < min_observations:
                continue
            
            # Test both directions: ref vs asset and asset vs ref
            # Direction 1: ref as reference, asset as target
            result1 = cointegration_with_reference(
                ref_aligned,
                asset_aligned,
                eg_alpha=eg_alpha,
                adf_alpha=adf_alpha,
                kss_critical_10pct=kss_critical,
                use_intercept=use_intercept
            )
            
            if result1 is not None:
                results.append({
                    'reference': ref_symbol,
                    'asset': asset_symbol,
                    'beta': result1.beta,
                    'eg_pvalue': result1.eg_pvalue,
                    'adf_pvalue': result1.adf_pvalue,
                    'kss_stat': result1.kss_stat,
                    'observations': len(result1.spread)
                })
            
            # Direction 2: asset as reference, ref as target
            result2 = cointegration_with_reference(
                asset_aligned,
                ref_aligned,
                eg_alpha=eg_alpha,
                adf_alpha=adf_alpha,
                kss_critical_10pct=kss_critical,
                use_intercept=use_intercept
            )
            
            if result2 is not None:
                results.append({
                    'reference': asset_symbol,
                    'asset': ref_symbol,
                    'beta': result2.beta,
                    'eg_pvalue': result2.eg_pvalue,
                    'adf_pvalue': result2.adf_pvalue,
                    'kss_stat': result2.kss_stat,
                    'observations': len(result2.spread)
                })
    
    print(f"   Checked all {checked} pairs.\n")
    return results

def generate_report(results: list[dict], output_file: str):
    """Generate a detailed report of cointegrated pairs."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    # Generate text report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("COINTEGRATED PAIRS ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Total pairs found: {len(results)}")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    if len(results) == 0:
        report_lines.append("No cointegrated pairs found matching the criteria.")
    else:
        # Summary statistics
        report_lines.append("SUMMARY STATISTICS")
        report_lines.append("-" * 80)
        report_lines.append(f"Total cointegrated pairs: {len(results)}")
        report_lines.append(f"Average EG p-value: {df['eg_pvalue'].mean():.4f}")
        report_lines.append(f"Average ADF p-value: {df['adf_pvalue'].mean():.4f}")
        report_lines.append(f"Average KSS statistic: {df['kss_stat'].mean():.4f}")
        report_lines.append(f"Average observations: {df['observations'].mean():.1f}")
        report_lines.append("")
        
        # Top pairs by statistical strength
        report_lines.append("TOP 20 PAIRS BY STATISTICAL STRENGTH")
        report_lines.append("(Sorted by combined p-value score)")
        report_lines.append("-" * 80)
        
        # Create a composite score (lower is better)
        df['score'] = df['eg_pvalue'] + df['adf_pvalue'] + (df['kss_stat'].abs() / 10)
        df_sorted = df.sort_values('score')
        
        report_lines.append(f"{'Rank':<6} {'Reference':<20} {'Asset':<20} {'Beta':<10} {'EG p-val':<10} {'ADF p-val':<10} {'KSS stat':<10}")
        report_lines.append("-" * 80)
        
        for idx, row in df_sorted.head(20).iterrows():
            report_lines.append(
                f"{idx+1:<6} {row['reference']:<20} {row['asset']:<20} "
                f"{row['beta']:<10.4f} {row['eg_pvalue']:<10.4f} {row['adf_pvalue']:<10.4f} {row['kss_stat']:<10.4f}"
            )
        
        report_lines.append("")
        report_lines.append("")
        
        # All pairs detailed listing
        report_lines.append("ALL COINTEGRATED PAIRS")
        report_lines.append("-" * 80)
        
        for idx, row in df_sorted.iterrows():
            report_lines.append(f"\nPair #{idx+1}: {row['reference']} vs {row['asset']}")
            report_lines.append(f"  Beta coefficient: {row['beta']:.6f}")
            report_lines.append(f"  Engle-Granger p-value: {row['eg_pvalue']:.6f}")
            report_lines.append(f"  ADF p-value: {row['adf_pvalue']:.6f}")
            report_lines.append(f"  KSS t-statistic: {row['kss_stat']:.6f}")
            report_lines.append(f"  Observations: {row['observations']}")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 80)
    
    # Write text report
    txt_file = f"{output_file}_{timestamp}.txt"
    with open(txt_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"📄 Text report saved to: {txt_file}")
    
    # Save CSV for further analysis
    if len(results) > 0:
        csv_file = f"{output_file}_{timestamp}.csv"
        df_sorted.to_csv(csv_file, index=False)
        print(f"📊 CSV data saved to: {csv_file}")
    
    return txt_file, df

def main():
    """Main execution function."""
    print("=" * 80)
    print("🔬 COINTEGRATION ANALYSIS: complete_data")
    print("=" * 80)
    print()
    
    # Configuration
    data_dir = "data/complete_data"
    output_file = "reports/cointegrated_pairs"
    
    # Load data
    print("📂 Loading data from complete_data directory...")
    print()
    data = load_data(data_dir)
    
    if len(data) < 2:
        print("❌ Error: Need at least 2 assets to find pairs")
        return
    
    print(f"\n✓ Loaded {len(data)} assets")
    
    # Find cointegrated pairs
    results = find_all_cointegrated_pairs(
        data,
        eg_alpha=0.10,
        adf_alpha=0.10,
        kss_critical=-1.92,
        use_intercept=False,
        min_observations=200
    )
    
    # Generate report
    print(f"📊 Generating report...")
    txt_file, df = generate_report(results, output_file)
    
    print()
    print("=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Found {len(results)} cointegrated pairs")
    print(f"Report saved to: {txt_file}")
    print("=" * 80)
    
    # Print quick summary
    if len(results) > 0:
        print("\n📈 Top 5 Strongest Pairs:")
        print("-" * 80)
        df_sorted = df.sort_values('score')
        for i, (idx, row) in enumerate(df_sorted.head(5).iterrows()):
            print(f"{i+1}. {row['reference']} vs {row['asset']}")
            print(f"   EG={row['eg_pvalue']:.4f}, ADF={row['adf_pvalue']:.4f}, KSS={row['kss_stat']:.4f}")

if __name__ == "__main__":
    main()
