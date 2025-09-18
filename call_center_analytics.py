import os
import re
import glob
from pathlib import Path
import json
from datetime import datetime
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from collections import defaultdict, Counter
import statistics

#Get all audio files from the specified folder to process.
def get_audio_files(audio_folder):
    """
    Get all supported audio files from the audio folder
    """
    supported_formats = ['*.wav', '*.mp3', '*.flac', '*.m4a', '*.ogg']
    audio_files = []
    
    for format_pattern in supported_formats:
        audio_files.extend(glob.glob(os.path.join(audio_folder, format_pattern)))
        audio_files.extend(glob.glob(os.path.join(audio_folder, format_pattern.upper())))
    
    return sorted(audio_files)

#Use ElevenLabs API to transcribe audio files with diarization and word-level timestamps.
def transcribe_audio_file(client, file_path):
    """
    Transcribe a single audio file using ElevenLabs API
    """
    try:
        print(f"Processing: {os.path.basename(file_path)}")
        
        with open(file_path, "rb") as audio_file:
            transcription = client.speech_to_text.convert(
                file=audio_file,
                model_id="scribe_v1",
                tag_audio_events=True,
                language_code="eng",
                diarize=True,
            )
        
        return transcription.words
    
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

#Process transcription words to analyze call structure and speaker turns.
def analyze_call_structure(transcription_words):
    """
    Analyze call flow and structure
    """
    turns = []
    current_speaker = None
    turn_start = None
    word_count = 0
    
    for word_obj in transcription_words:
        if word_obj.type == 'word':
            if current_speaker != word_obj.speaker_id:
                if current_speaker:
                    turns.append({
                        'speaker': current_speaker,
                        'start': turn_start,
                        'end': word_obj.start,
                        'word_count': word_count
                    })
                current_speaker = word_obj.speaker_id
                turn_start = word_obj.start
                word_count = 1
            else:
                word_count += 1
    
    # Add final turn
    if current_speaker and turns:
        final_word = [w for w in transcription_words if w.type == 'word'][-1]
        turns.append({
            'speaker': current_speaker,
            'start': turn_start,
            'end': final_word.end,
            'word_count': word_count
        })
    
    return turns

# This is a helper function to extract sentiment indicators. It is a basic implementation
# and can be expanded with more sophisticated NLP techniques if needed.
def extract_customer_sentiment_indicators(transcription_words):
    """
    Extract potential sentiment indicators from speech patterns
    """
    # Positive indicators
    positive_words = ['thanks', 'thank', 'great', 'good', 'perfect', 'excellent', 'wonderful', 'appreciate']
    
    # Negative indicators  
    negative_words = ['problem', 'issue', 'trouble', 'frustrated', 'angry', 'upset', 'disappointed', 
                     'terrible', 'awful', 'hate', 'annoyed', 'broken', 'wrong', 'bad']
    
    # Politeness indicators
    polite_words = ['please', 'sorry', 'excuse', 'pardon', 'thank', 'thanks']
    
    # Urgency indicators
    urgent_words = ['urgent', 'emergency', 'immediately', 'asap', 'quickly', 'fast', 'right now']
    
    sentiment_analysis = {
        'positive_indicators': [],
        'negative_indicators': [],
        'polite_indicators': [],
        'urgency_indicators': []
    }
    
    all_words = [w.text.lower().strip('.,!?') for w in transcription_words if w.type == 'word']
    
    for word in all_words:
        if word in positive_words:
            sentiment_analysis['positive_indicators'].append(word)
        if word in negative_words:
            sentiment_analysis['negative_indicators'].append(word)
        if word in polite_words:
            sentiment_analysis['polite_indicators'].append(word)
        if word in urgent_words:
            sentiment_analysis['urgency_indicators'].append(word)
    
    return sentiment_analysis

# Analyze speech recognition confidence levels. This is a basic implementation and can be expanded with more sophisticated NLP techniques if needed.
def analyze_speech_confidence(transcription_words):
    """
    Analyze speech recognition confidence levels
    """
    word_confidences = [w.logprob for w in transcription_words if w.type == 'word' and w.logprob is not None]
    
    if not word_confidences:
        return None
    
    return {
        'avg_confidence': statistics.mean(word_confidences),
        'min_confidence': min(word_confidences),
        'max_confidence': max(word_confidences),
        'low_confidence_words': len([c for c in word_confidences if c < -2.0]),
        'total_words': len(word_confidences)
    }

#This is a basic implementation and can be expanded with more sophisticated NLP techniques if needed.
def detect_call_outcome(transcription_words):
    """
    Detect potential call outcomes and resolution indicators
    """
    all_text = ' '.join([w.text for w in transcription_words if w.type == 'word']).lower()
    
    outcomes = {
        'resolved': False,
        'escalated': False,
        'callback_scheduled': False,
        'customer_satisfied': False,
        'technical_issue': False
    }
    
    if any(phrase in all_text for phrase in ['fixed', 'resolved', 'working', 'solved', 'better now']):
        outcomes['resolved'] = True
    
    if any(phrase in all_text for phrase in ['transfer', 'supervisor', 'manager', 'escalate']):
        outcomes['escalated'] = True
    
    if any(phrase in all_text for phrase in ['call back', 'callback', 'call you back', 'follow up']):
        outcomes['callback_scheduled'] = True
    
    if any(phrase in all_text for phrase in ['thank', 'great', 'perfect', 'excellent']):
        outcomes['customer_satisfied'] = True
    
    if any(phrase in all_text for phrase in ['internet', 'connection', 'router', 'computer', 'pc', 'network']):
        outcomes['technical_issue'] = True
    
    return outcomes

def extract_key_information(transcription_words):
    """
    Extract key business information like account numbers, names, etc.
    """
    all_text = ' '.join([w.text for w in transcription_words if w.type == 'word'])
    
    info = {
        'account_numbers': [],
        'phone_numbers': [],
        'names': [],
        'email_addresses': []
    }
    
    account_pattern = r'\b\d{6,}\b'
    info['account_numbers'] = re.findall(account_pattern, all_text)
    
    phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    info['phone_numbers'] = re.findall(phone_pattern, all_text)
    
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    info['email_addresses'] = re.findall(email_pattern, all_text)
    
    name_pattern = r'(?:my name is|I\'m|I am)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'
    names = re.findall(name_pattern, all_text, re.IGNORECASE)
    info['names'] = names
    
    return info

def calculate_interaction_metrics(transcription_words, turns):
    """
    Calculate various interaction quality metrics
    """
    if not turns:
        return {}
    
    agent_turns = [t for t in turns if t['speaker'] == 'speaker_0']
    customer_turns = [t for t in turns if t['speaker'] == 'speaker_1']
    
    metrics = {
        'total_turns': len(turns),
        'agent_turns': len(agent_turns),
        'customer_turns': len(customer_turns),
        'avg_turn_duration': statistics.mean([t['end'] - t['start'] for t in turns]),
        'agent_talk_ratio': 0,
        'customer_talk_ratio': 0,
        'turn_taking_frequency': 0,
        'longest_silence': 0
    }
    
    if turns:
        total_time = turns[-1]['end'] - turns[0]['start']
        agent_time = sum([t['end'] - t['start'] for t in agent_turns])
        customer_time = sum([t['end'] - t['start'] for t in customer_turns])
        
        metrics['agent_talk_ratio'] = agent_time / total_time if total_time > 0 else 0
        metrics['customer_talk_ratio'] = customer_time / total_time if total_time > 0 else 0
        metrics['turn_taking_frequency'] = len(turns) / (total_time / 60) if total_time > 0 else 0
        metrics['call_duration'] = total_time
    
    # Calculate silences between turns
    silences = []
    for i in range(1, len(turns)):
        silence = turns[i]['start'] - turns[i-1]['end']
        if silence > 0:
            silences.append(silence)
    
    if silences:
        metrics['longest_silence'] = max(silences)
        metrics['avg_silence'] = statistics.mean(silences)
    
    return metrics

def analyze_single_call(transcription_words, filename):
    """
    Analyze a single call and return structured results
    """
    turns = analyze_call_structure(transcription_words)
    sentiment = extract_customer_sentiment_indicators(transcription_words)
    confidence = analyze_speech_confidence(transcription_words)
    outcomes = detect_call_outcome(transcription_words)
    key_info = extract_key_information(transcription_words)
    metrics = calculate_interaction_metrics(transcription_words, turns)
    
    # Calculate sentiment score
    sentiment_score = len(sentiment['positive_indicators']) - len(sentiment['negative_indicators'])
    
    return {
        'filename': filename,
        'call_duration': metrics.get('call_duration', 0),
        'total_turns': metrics.get('total_turns', 0),
        'agent_talk_ratio': metrics.get('agent_talk_ratio', 0),
        'customer_talk_ratio': metrics.get('customer_talk_ratio', 0),
        'sentiment_score': sentiment_score,
        'positive_indicators': len(sentiment['positive_indicators']),
        'negative_indicators': len(sentiment['negative_indicators']),
        'resolved': outcomes['resolved'],
        'escalated': outcomes['escalated'],
        'customer_satisfied': outcomes['customer_satisfied'],
        'technical_issue': outcomes['technical_issue'],
        'avg_confidence': confidence['avg_confidence'] if confidence else 0,
        'low_confidence_percentage': (confidence['low_confidence_words'] / confidence['total_words'] * 100) if confidence else 0,
        'account_numbers_found': len(key_info['account_numbers']),
        'names_found': len(key_info['names']),
        'turn_taking_frequency': metrics.get('turn_taking_frequency', 0),
        'longest_silence': metrics.get('longest_silence', 0),
        'detailed_analysis': {
            'turns': turns,
            'sentiment': sentiment,
            'confidence': confidence,
            'outcomes': outcomes,
            'key_info': key_info,
            'metrics': metrics
        }
    }

def generate_aggregated_report(all_results):
    """
    Generate aggregated analytics report across all calls
    """
    if not all_results:
        print("No calls to analyze")
        return
    
    print("="*80)
    print("AGGREGATED CALL CENTER ANALYTICS REPORT")
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Calls Analyzed: {len(all_results)}")
    print("="*80)
    
    # Summary Statistics
    print("\nCALL VOLUME SUMMARY")
    print("-" * 50)
    total_duration = sum([r['call_duration'] for r in all_results])
    avg_duration = statistics.mean([r['call_duration'] for r in all_results])
    
    print(f"Total Call Time: {total_duration/60:.1f} minutes ({total_duration/3600:.1f} hours)")
    print(f"Average Call Duration: {avg_duration:.1f} seconds ({avg_duration/60:.1f} minutes)")
    print(f"Shortest Call: {min([r['call_duration'] for r in all_results]):.1f} seconds")
    print(f"Longest Call: {max([r['call_duration'] for r in all_results]):.1f} seconds")
    
    # Resolution and Outcome Statistics
    print("\nCALL OUTCOMES")
    print("-" * 50)
    resolved_calls = sum([1 for r in all_results if r['resolved']])
    escalated_calls = sum([1 for r in all_results if r['escalated']])
    satisfied_customers = sum([1 for r in all_results if r['customer_satisfied']])
    technical_issues = sum([1 for r in all_results if r['technical_issue']])
    
    print(f"Resolved Calls: {resolved_calls}/{len(all_results)} ({resolved_calls/len(all_results)*100:.1f}%)")
    print(f"Escalated Calls: {escalated_calls}/{len(all_results)} ({escalated_calls/len(all_results)*100:.1f}%)")
    print(f"Customer Satisfaction: {satisfied_customers}/{len(all_results)} ({satisfied_customers/len(all_results)*100:.1f}%)")
    print(f"Technical Issues: {technical_issues}/{len(all_results)} ({technical_issues/len(all_results)*100:.1f}%)")
    
    # Sentiment Analysis
    print("\nSENTIMENT ANALYSIS")
    print("-" * 50)
    avg_sentiment = statistics.mean([r['sentiment_score'] for r in all_results])
    positive_calls = sum([1 for r in all_results if r['sentiment_score'] > 0])
    negative_calls = sum([1 for r in all_results if r['sentiment_score'] < 0])
    neutral_calls = len(all_results) - positive_calls - negative_calls
    
    print(f"Average Sentiment Score: {avg_sentiment:.2f}")
    print(f"Positive Sentiment: {positive_calls}/{len(all_results)} ({positive_calls/len(all_results)*100:.1f}%)")
    print(f"Negative Sentiment: {negative_calls}/{len(all_results)} ({negative_calls/len(all_results)*100:.1f}%)")
    print(f"Neutral Sentiment: {neutral_calls}/{len(all_results)} ({neutral_calls/len(all_results)*100:.1f}%)")
    
    # Agent Performance
    print("\nAGENT PERFORMANCE METRICS")
    print("-" * 50)
    avg_agent_ratio = statistics.mean([r['agent_talk_ratio'] for r in all_results])
    avg_customer_ratio = statistics.mean([r['customer_talk_ratio'] for r in all_results])
    avg_turns = statistics.mean([r['total_turns'] for r in all_results])
    
    print(f"Average Agent Talk Time: {avg_agent_ratio*100:.1f}%")
    print(f"Average Customer Talk Time: {avg_customer_ratio*100:.1f}%")
    print(f"Average Turns per Call: {avg_turns:.1f}")
    
    # Audio Quality
    print("\nAUDIO QUALITY METRICS")
    print("-" * 50)
    confidences = [r['avg_confidence'] for r in all_results if r['avg_confidence'] != 0]
    if confidences:
        avg_confidence = statistics.mean(confidences)
        avg_low_confidence = statistics.mean([r['low_confidence_percentage'] for r in all_results])
        
        print(f"Average Speech Confidence: {avg_confidence:.2f}")
        print(f"Average Low Confidence Words: {avg_low_confidence:.1f}%")
        
        quality_rating = "Excellent" if avg_confidence > -1 else "Good" if avg_confidence > -1.5 else "Fair" if avg_confidence > -2 else "Poor"
        print(f"Overall Audio Quality Rating: {quality_rating}")
    
    # Top Issues and Patterns
    print("\nTOP FINDINGS")
    print("-" * 50)
    
    # Longest calls
    longest_calls = sorted(all_results, key=lambda x: x['call_duration'], reverse=True)[:3]
    print(f"Longest Calls:")
    for i, call in enumerate(longest_calls, 1):
        print(f"  {i}. {call['filename']}: {call['call_duration']/60:.1f} minutes")
    
    # Most negative sentiment
    most_negative = sorted(all_results, key=lambda x: x['sentiment_score'])[:3]
    print(f"\nCalls with Most Negative Sentiment:")
    for i, call in enumerate(most_negative, 1):
        print(f"  {i}. {call['filename']}: Score {call['sentiment_score']}")
    
    # Escalated calls
    escalated = [r for r in all_results if r['escalated']]
    if escalated:
        print(f"\nEscalated Calls:")
        for call in escalated:
            print(f"  - {call['filename']}")
    
    # Recommendations
    print("\nRECOMMendations")
    print("-" * 50)
    recommendations = []
    
    if avg_agent_ratio > 0.7:
        recommendations.append("Agents are dominating conversations - consider training on customer engagement")
    
    if negative_calls / len(all_results) > 0.3:
        recommendations.append("High negative sentiment detected - review customer service protocols")
    
    if escalated_calls / len(all_results) > 0.2:
        recommendations.append("High escalation rate - investigate common issues and agent training needs")
    
    if resolved_calls / len(all_results) < 0.7:
        recommendations.append("Low resolution rate - review troubleshooting procedures")
    
    if confidences and avg_confidence < -2:
        recommendations.append("Poor audio quality across calls - check recording equipment and environment")
    
    if not recommendations:
        recommendations.append("Overall call quality appears good - continue current practices")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    return {
        'total_calls': len(all_results),
        'avg_duration': avg_duration,
        'resolution_rate': resolved_calls / len(all_results),
        'escalation_rate': escalated_calls / len(all_results),
        'satisfaction_rate': satisfied_customers / len(all_results),
        'avg_sentiment': avg_sentiment,
        'avg_agent_ratio': avg_agent_ratio,
        'recommendations': recommendations
    }

def save_results_to_json(all_results, aggregated_results, output_file='call_analytics_results.json'):
    """
    Save all results to a JSON file for further analysis
    """
    output_data = {
        'analysis_timestamp': datetime.now().isoformat(),
        'aggregated_results': aggregated_results,
        'individual_calls': all_results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed results saved to: {output_file}")

def main():
    load_dotenv()
    
    # Configuration
    audio_folder = "audio"  # Folder containing audio files
    output_file = "call_analytics_results.json"
    
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("ELEVENLABS_API_KEY environment variable not set.")
        return
    
    client = ElevenLabs(api_key=api_key)
    
    # Check if audio folder exists
    if not os.path.exists(audio_folder):
        print(f"Audio folder '{audio_folder}' not found. Please create it and add audio files.")
        return
    
    # Get all audio files
    audio_files = get_audio_files(audio_folder)
    if not audio_files:
        print(f"No supported audio files found in '{audio_folder}' folder.")
        print("Supported formats: WAV, MP3, FLAC, M4A, OGG")
        return
    
    print(f"Found {len(audio_files)} audio files to process...")
    
    all_results = []
    
    # Process each audio file
    for i, file_path in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}] Processing: {os.path.basename(file_path)}")
        
        transcription_words = transcribe_audio_file(client, file_path)
        if transcription_words:
            result = analyze_single_call(transcription_words, os.path.basename(file_path))
            all_results.append(result)
            print(f"  - Duration: {result['call_duration']:.1f}s, Sentiment: {result['sentiment_score']}, Resolved: {result['resolved']}")
        else:
            print(f"  - Failed to process {os.path.basename(file_path)}")
    
    if all_results:
        # Generate aggregated report
        print(f"\n{'-'*80}")
        print("PROCESSING COMPLETE - GENERATING AGGREGATED REPORT")
        print(f"{'-'*80}")
        
        aggregated_results = generate_aggregated_report(all_results)
        
        # Save results to JSON file
        save_results_to_json(all_results, aggregated_results, output_file)
        
        print(f"\nAnalysis complete! Processed {len(all_results)} calls successfully.")
    else:
        print("No calls were successfully processed.")

if __name__ == "__main__":
    main()