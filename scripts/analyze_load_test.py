#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def load_jtl_file(file_path):
    """Load JMeter JTL file into a pandas DataFrame."""
    df = pd.read_csv(file_path)
    df['timeStamp'] = pd.to_datetime(df['timeStamp'], unit='ms')
    return df

def calculate_metrics(df):
    """Calculate key performance metrics."""
    metrics = {
        'total_requests': len(df),
        'successful_requests': len(df[df['success'] == True]),
        'failed_requests': len(df[df['success'] == False]),
        'average_response_time': df['elapsed'].mean(),
        'median_response_time': df['elapsed'].median(),
        'p95_response_time': df['elapsed'].quantile(0.95),
        'p99_response_time': df['elapsed'].quantile(0.99),
        'min_response_time': df['elapsed'].min(),
        'max_response_time': df['elapsed'].max(),
        'requests_per_second': len(df) / (df['timeStamp'].max() - df['timeStamp'].min()).total_seconds(),
        'error_rate': len(df[df['success'] == False]) / len(df) * 100
    }
    return metrics

def analyze_response_times_by_endpoint(df):
    """Analyze response times grouped by endpoint."""
    endpoint_metrics = df.groupby('label').agg({
        'elapsed': ['count', 'mean', 'median', 'std', lambda x: np.percentile(x, 95)],
        'success': 'mean'
    })
    endpoint_metrics.columns = ['count', 'mean_rt', 'median_rt', 'std_rt', 'p95_rt', 'success_rate']
    return endpoint_metrics

def plot_response_time_distribution(df, output_dir):
    """Create response time distribution plots."""
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='label', y='elapsed', data=df)
    plt.xticks(rotation=45)
    plt.title('Response Time Distribution by Endpoint')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/response_time_distribution.png')
    plt.close()

def plot_requests_over_time(df, output_dir):
    """Plot number of requests and response times over time."""
    df_time = df.set_index('timeStamp')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Requests per second
    df_time.resample('1S').size().plot(ax=ax1)
    ax1.set_title('Requests per Second')
    
    # Response times
    df_time['elapsed'].plot(ax=ax2, style='.')
    ax2.set_title('Response Times')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/requests_over_time.png')
    plt.close()

def validate_performance_requirements(metrics):
    """Validate if performance meets requirements."""
    requirements = {
        'max_p95_response_time': 500,  # 500ms
        'min_success_rate': 99.0,      # 99%
        'min_requests_per_second': 100  # 100 RPS
    }
    
    validations = {
        'p95_response_time': metrics['p95_response_time'] <= requirements['max_p95_response_time'],
        'success_rate': (metrics['successful_requests'] / metrics['total_requests'] * 100) >= requirements['min_success_rate'],
        'requests_per_second': metrics['requests_per_second'] >= requirements['min_requests_per_second']
    }
    
    return validations

def generate_report(metrics, endpoint_metrics, validations, output_file):
    """Generate a detailed performance report."""
    with open(output_file, 'w') as f:
        f.write('US Tree Dashboard Load Test Report\n')
        f.write('=' * 40 + '\n\n')
        
        f.write('Overall Metrics:\n')
        f.write('-' * 20 + '\n')
        for metric, value in metrics.items():
            f.write(f'{metric}: {value:.2f}\n')
        
        f.write('\nEndpoint Performance:\n')
        f.write('-' * 20 + '\n')
        f.write(endpoint_metrics.to_string())
        
        f.write('\n\nPerformance Validation:\n')
        f.write('-' * 20 + '\n')
        for check, passed in validations.items():
            status = 'PASSED' if passed else 'FAILED'
            f.write(f'{check}: {status}\n')

def main(jtl_file):
    # Create output directory for artifacts
    output_dir = 'load_test_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and analyze data
    df = load_jtl_file(jtl_file)
    metrics = calculate_metrics(df)
    endpoint_metrics = analyze_response_times_by_endpoint(df)
    
    # Generate visualizations
    plot_response_time_distribution(df, output_dir)
    plot_requests_over_time(df, output_dir)
    
    # Validate performance
    validations = validate_performance_requirements(metrics)
    
    # Generate report
    generate_report(metrics, endpoint_metrics, validations, f'{output_dir}/load_test_report.txt')
    
    # Exit with error if any validation failed
    if not all(validations.values()):
        print('Performance requirements not met. Check load_test_report.txt for details.')
        sys.exit(1)
    else:
        print('All performance requirements met. Check load_test_report.txt for details.')
        sys.exit(0)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python analyze_load_test.py <jtl_file>')
        sys.exit(1)
    
    main(sys.argv[1])