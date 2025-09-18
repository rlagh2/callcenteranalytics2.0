import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import os
from pathlib import Path

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_analytics_data(json_file='call_analytics_results.json'):
    """
    Load analytics data from JSON file
    """
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"Analytics file {json_file} not found. Run the analytics script first.")
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data

def create_call_duration_analysis(individual_calls, save_dir='analytics_charts'):
    """
    Create call duration analysis charts
    """
    durations = [call['call_duration'] for call in individual_calls]
    duration_minutes = [d/60 for d in durations]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram of call durations
    ax1.hist(duration_minutes, bins=20, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Call Duration (minutes)')
    ax1.set_ylabel('Number of Calls')
    ax1.set_title('Distribution of Call Durations')
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2.boxplot(duration_minutes, vert=True)
    ax2.set_ylabel('Call Duration (minutes)')
    ax2.set_title('Call Duration Statistics')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f'Mean: {np.mean(duration_minutes):.1f} min\nMedian: {np.median(duration_minutes):.1f} min\nStd: {np.std(duration_minutes):.1f} min'
    ax2.text(0.7, max(duration_minutes)*0.8, stats_text, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/call_duration_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_outcome_analysis(individual_calls, save_dir='analytics_charts'):
    """
    Create call outcome analysis charts
    """
    outcomes = {
        'Resolved': sum([1 for call in individual_calls if call['resolved']]),
        'Escalated': sum([1 for call in individual_calls if call['escalated']]),
        'Customer Satisfied': sum([1 for call in individual_calls if call['customer_satisfied']]),
        'Technical Issue': sum([1 for call in individual_calls if call['technical_issue']])
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Pie chart for resolutions
    resolution_data = {
        'Resolved': outcomes['Resolved'],
        'Not Resolved': len(individual_calls) - outcomes['Resolved']
    }
    
    colors = ['#2ecc71', '#e74c3c']
    wedges, texts, autotexts = ax1.pie(resolution_data.values(), labels=resolution_data.keys(), 
                                       autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title('Call Resolution Rate')
    
    # Bar chart for all outcomes
    ax2.bar(outcomes.keys(), outcomes.values(), 
            color=['#3498db', '#f39c12', '#9b59b6', '#1abc9c'])
    ax2.set_ylabel('Number of Calls')
    ax2.set_title('Call Outcomes Summary')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add percentage labels on bars
    total_calls = len(individual_calls)
    for i, (key, value) in enumerate(outcomes.items()):
        percentage = (value / total_calls) * 100
        ax2.text(i, value + 0.1, f'{percentage:.1f}%', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/call_outcomes.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_sentiment_analysis(individual_calls, save_dir='analytics_charts'):
    """
    Create sentiment analysis visualizations
    """
    sentiment_scores = [call['sentiment_score'] for call in individual_calls]
    positive_indicators = [call['positive_indicators'] for call in individual_calls]
    negative_indicators = [call['negative_indicators'] for call in individual_calls]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Sentiment score distribution
    ax1.hist(sentiment_scores, bins=15, alpha=0.7, edgecolor='black', color='skyblue')
    ax1.axvline(np.mean(sentiment_scores), color='red', linestyle='--', 
               label=f'Mean: {np.mean(sentiment_scores):.2f}')
    ax1.set_xlabel('Sentiment Score')
    ax1.set_ylabel('Number of Calls')
    ax1.set_title('Distribution of Sentiment Scores')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Sentiment categories pie chart
    positive_calls = sum([1 for score in sentiment_scores if score > 0])
    negative_calls = sum([1 for score in sentiment_scores if score < 0])
    neutral_calls = len(sentiment_scores) - positive_calls - negative_calls
    
    sentiment_categories = {
        'Positive': positive_calls,
        'Negative': negative_calls,
        'Neutral': neutral_calls
    }
    
    colors = ['#2ecc71', '#e74c3c', '#95a5a6']
    ax2.pie(sentiment_categories.values(), labels=sentiment_categories.keys(), 
           autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('Sentiment Distribution')
    
    # Positive vs Negative indicators scatter
    ax3.scatter(positive_indicators, negative_indicators, alpha=0.6, s=60)
    ax3.set_xlabel('Positive Indicators Count')
    ax3.set_ylabel('Negative Indicators Count')
    ax3.set_title('Positive vs Negative Sentiment Indicators')
    ax3.grid(True, alpha=0.3)
    
    # Add trend line
    if len(positive_indicators) > 1:
        z = np.polyfit(positive_indicators, negative_indicators, 1)
        p = np.poly1d(z)
        ax3.plot(positive_indicators, p(positive_indicators), "r--", alpha=0.8)
    
    # Sentiment over time (by call index as proxy)
    call_indices = list(range(1, len(sentiment_scores) + 1))
    ax4.plot(call_indices, sentiment_scores, marker='o', markersize=4, alpha=0.7)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.set_xlabel('Call Index')
    ax4.set_ylabel('Sentiment Score')
    ax4.set_title('Sentiment Trend Across Calls')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/sentiment_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_agent_performance_analysis(individual_calls, save_dir='analytics_charts'):
    """
    Create agent performance analysis charts
    """
    agent_ratios = [call['agent_talk_ratio'] * 100 for call in individual_calls]
    customer_ratios = [call['customer_talk_ratio'] * 100 for call in individual_calls]
    total_turns = [call['total_turns'] for call in individual_calls]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Agent vs Customer talk time
    ax1.scatter(agent_ratios, customer_ratios, alpha=0.6, s=60)
    ax1.set_xlabel('Agent Talk Time (%)')
    ax1.set_ylabel('Customer Talk Time (%)')
    ax1.set_title('Agent vs Customer Talk Time Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Add diagonal line (ideal balance would be around 50-50)
    ax1.plot([0, 100], [100, 0], 'r--', alpha=0.5, label='Perfect Balance Line')
    ax1.legend()
    
    # Agent talk ratio distribution
    ax2.hist(agent_ratios, bins=15, alpha=0.7, edgecolor='black', color='lightcoral')
    ax2.axvline(np.mean(agent_ratios), color='red', linestyle='--', 
               label=f'Mean: {np.mean(agent_ratios):.1f}%')
    ax2.set_xlabel('Agent Talk Time (%)')
    ax2.set_ylabel('Number of Calls')
    ax2.set_title('Distribution of Agent Talk Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Turn frequency analysis
    ax3.hist(total_turns, bins=15, alpha=0.7, edgecolor='black', color='lightgreen')
    ax3.axvline(np.mean(total_turns), color='red', linestyle='--', 
               label=f'Mean: {np.mean(total_turns):.1f} turns')
    ax3.set_xlabel('Total Turns per Call')
    ax3.set_ylabel('Number of Calls')
    ax3.set_title('Distribution of Conversation Turns')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Agent performance quadrants
    ax4.scatter(agent_ratios, total_turns, alpha=0.6, s=60, c=customer_ratios, 
               cmap='viridis')
    ax4.set_xlabel('Agent Talk Time (%)')
    ax4.set_ylabel('Total Turns')
    ax4.set_title('Agent Performance Matrix')
    cbar = plt.colorbar(ax4.collections[0], ax=ax4)
    cbar.set_label('Customer Talk Time (%)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/agent_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_audio_quality_analysis(individual_calls, save_dir='analytics_charts'):
    """
    Create audio quality analysis charts
    """
    # Filter out calls with no confidence data
    calls_with_confidence = [call for call in individual_calls if call['avg_confidence'] != 0]
    
    if not calls_with_confidence:
        print("No audio confidence data available for visualization.")
        return
    
    confidence_scores = [call['avg_confidence'] for call in calls_with_confidence]
    low_confidence_percentages = [call['low_confidence_percentage'] for call in calls_with_confidence]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Confidence score distribution
    ax1.hist(confidence_scores, bins=15, alpha=0.7, edgecolor='black', color='gold')
    ax1.axvline(np.mean(confidence_scores), color='red', linestyle='--', 
               label=f'Mean: {np.mean(confidence_scores):.2f}')
    ax1.set_xlabel('Average Confidence Score')
    ax1.set_ylabel('Number of Calls')
    ax1.set_title('Distribution of Speech Recognition Confidence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Quality categories
    excellent = sum([1 for score in confidence_scores if score > -1])
    good = sum([1 for score in confidence_scores if -1.5 < score <= -1])
    fair = sum([1 for score in confidence_scores if -2 < score <= -1.5])
    poor = sum([1 for score in confidence_scores if score <= -2])
    
    quality_data = {
        'Excellent': excellent,
        'Good': good,
        'Fair': fair,
        'Poor': poor
    }
    
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
    ax2.pie([v for v in quality_data.values() if v > 0], 
           labels=[k for k, v in quality_data.items() if v > 0], 
           autopct='%1.1f%%', colors=colors[:len([v for v in quality_data.values() if v > 0])], 
           startangle=90)
    ax2.set_title('Audio Quality Distribution')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/audio_quality.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_correlation_heatmap(individual_calls, save_dir='analytics_charts'):
    """
    Create correlation heatmap of key metrics
    """
    # Prepare data for correlation analysis
    df_data = {
        'Call Duration': [call['call_duration'] for call in individual_calls],
        'Agent Talk Ratio': [call['agent_talk_ratio'] for call in individual_calls],
        'Total Turns': [call['total_turns'] for call in individual_calls],
        'Sentiment Score': [call['sentiment_score'] for call in individual_calls],
        'Positive Indicators': [call['positive_indicators'] for call in individual_calls],
        'Negative Indicators': [call['negative_indicators'] for call in individual_calls],
        'Resolved': [int(call['resolved']) for call in individual_calls],
        'Customer Satisfied': [int(call['customer_satisfied']) for call in individual_calls],
        'Escalated': [int(call['escalated']) for call in individual_calls]
    }
    
    # Add confidence data if available
    confidence_data = [call['avg_confidence'] for call in individual_calls if call['avg_confidence'] != 0]
    if len(confidence_data) == len(individual_calls):
        df_data['Avg Confidence'] = confidence_data
    
    df = pd.DataFrame(df_data)
    correlation_matrix = df.corr()
    
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f', cbar_kws={"shrink": .8})
    plt.title('Correlation Matrix of Call Metrics')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_comprehensive_dashboard(individual_calls, aggregated_results, save_dir='analytics_charts'):
    """
    Create a comprehensive dashboard summary
    """
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1, 1])
    
    # Key metrics summary
    ax1 = fig.add_subplot(gs[0, :2])
    metrics = [
        f"Total Calls: {aggregated_results['total_calls']}",
        f"Avg Duration: {aggregated_results['avg_duration']/60:.1f} min",
        f"Resolution Rate: {aggregated_results['resolution_rate']*100:.1f}%",
        f"Escalation Rate: {aggregated_results['escalation_rate']*100:.1f}%",
        f"Satisfaction Rate: {aggregated_results['satisfaction_rate']*100:.1f}%",
        f"Avg Sentiment: {aggregated_results['avg_sentiment']:.2f}"
    ]
    
    ax1.text(0.1, 0.8, "KEY PERFORMANCE INDICATORS", fontsize=16, fontweight='bold')
    for i, metric in enumerate(metrics):
        ax1.text(0.1, 0.7 - i*0.1, metric, fontsize=12)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    # Call outcomes pie chart
    ax2 = fig.add_subplot(gs[0, 2:])
    resolved = sum([1 for call in individual_calls if call['resolved']])
    unresolved = len(individual_calls) - resolved
    ax2.pie([resolved, unresolved], labels=['Resolved', 'Unresolved'], 
           autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'])
    ax2.set_title('Call Resolution Overview')
    
    # Sentiment distribution
    ax3 = fig.add_subplot(gs[1, 0])
    sentiment_scores = [call['sentiment_score'] for call in individual_calls]
    ax3.hist(sentiment_scores, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.set_title('Sentiment Distribution')
    ax3.set_xlabel('Sentiment Score')
    
    # Agent talk ratio
    ax4 = fig.add_subplot(gs[1, 1])
    agent_ratios = [call['agent_talk_ratio'] * 100 for call in individual_calls]
    ax4.hist(agent_ratios, bins=10, alpha=0.7, color='lightcoral', edgecolor='black')
    ax4.set_title('Agent Talk Time %')
    ax4.set_xlabel('Agent Talk Ratio (%)')
    
    # Call duration
    ax5 = fig.add_subplot(gs[1, 2])
    durations = [call['call_duration']/60 for call in individual_calls]
    ax5.hist(durations, bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
    ax5.set_title('Call Duration (min)')
    ax5.set_xlabel('Duration (minutes)')
    
    # Total turns
    ax6 = fig.add_subplot(gs[1, 3])
    turns = [call['total_turns'] for call in individual_calls]
    ax6.hist(turns, bins=10, alpha=0.7, color='gold', edgecolor='black')
    ax6.set_title('Conversation Turns')
    ax6.set_xlabel('Total Turns')
    
    # Recommendations
    ax7 = fig.add_subplot(gs[2, :])
    ax7.text(0.1, 0.9, "RECOMMENDATIONS", fontsize=14, fontweight='bold')
    for i, rec in enumerate(aggregated_results['recommendations'][:6]):  # Show max 6 recommendations
        ax7.text(0.1, 0.8 - i*0.12, f"{i+1}. {rec}", fontsize=10, wrap=True)
    ax7.set_xlim(0, 1)
    ax7.set_ylim(0, 1)
    ax7.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_all_visualizations(json_file='call_analytics_results.json', save_dir='analytics_charts'):
    """
    Generate all visualization charts from the analytics JSON file
    """
    # Create save directory if it doesn't exist
    Path(save_dir).mkdir(exist_ok=True)
    
    # Load data
    print("Loading analytics data...")
    data = load_analytics_data(json_file)
    individual_calls = data['individual_calls']
    aggregated_results = data['aggregated_results']
    
    print(f"Generating visualizations for {len(individual_calls)} calls...")
    
    # Generate all charts
    print("Creating call duration analysis...")
    create_call_duration_analysis(individual_calls, save_dir)
    
    print("Creating outcome analysis...")
    create_outcome_analysis(individual_calls, save_dir)
    
    print("Creating sentiment analysis...")
    create_sentiment_analysis(individual_calls, save_dir)
    
    print("Creating agent performance analysis...")
    create_agent_performance_analysis(individual_calls, save_dir)
    
    print("Creating audio quality analysis...")
    create_audio_quality_analysis(individual_calls, save_dir)
    
    print("Creating correlation heatmap...")
    create_correlation_heatmap(individual_calls, save_dir)
    
    print("Creating comprehensive dashboard...")
    create_comprehensive_dashboard(individual_calls, aggregated_results, save_dir)
    
    print(f"\nAll visualizations saved to '{save_dir}' folder:")
    print("- call_duration_analysis.png")
    print("- call_outcomes.png") 
    print("- sentiment_analysis.png")
    print("- agent_performance.png")
    print("- audio_quality.png")
    print("- correlation_heatmap.png")
    print("- comprehensive_dashboard.png")

if __name__ == "__main__":
    # Generate all visualizations
    generate_all_visualizations()
    
    # Example of generating individual charts:
    # data = load_analytics_data('call_analytics_results.json')
    # create_sentiment_analysis(data['individual_calls'])