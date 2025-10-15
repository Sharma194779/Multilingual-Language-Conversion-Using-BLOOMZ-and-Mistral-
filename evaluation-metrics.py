# Evaluation Metrics Implementation
# BLEU and chrF evaluation for multilingual translation

import sacrebleu
from evaluate import load
import numpy as np
from typing import List, Dict, Tuple
import pandas as pd
from collections import defaultdict

class DetailedEvaluationMetrics:
    """Comprehensive evaluation metrics for translation quality assessment"""
    
    def __init__(self):
        """Initialize evaluation metrics"""
        try:
            self.chrf_metric = load("chrf")
            self.rouge_metric = load("rouge")
        except Exception as e:
            print(f"Warning: Could not load some metrics: {e}")
            self.chrf_metric = None
            self.rouge_metric = None
    
    def calculate_bleu_detailed(self, predictions: List[str], references: List[List[str]]) -> Dict:
        """Calculate detailed BLEU scores including individual n-gram scores"""
        results = {}
        
        # Corpus-level BLEU
        corpus_bleu = sacrebleu.corpus_bleu(predictions, list(zip(*references)))
        results['corpus_bleu'] = {
            'score': corpus_bleu.score,
            'bp': corpus_bleu.bp,  # Brevity penalty
            'ratio': corpus_bleu.ratio,
            'hyp_len': corpus_bleu.hyp_len,
            'ref_len': corpus_bleu.ref_len,
            'precisions': corpus_bleu.precisions
        }
        
        # Sentence-level BLEU scores
        sentence_bleus = []
        for pred, ref_list in zip(predictions, references):
            sent_bleu = sacrebleu.sentence_bleu(pred, ref_list)
            sentence_bleus.append(sent_bleu.score)
        
        results['sentence_bleu'] = {
            'scores': sentence_bleus,
            'mean': np.mean(sentence_bleus),
            'std': np.std(sentence_bleus),
            'min': np.min(sentence_bleus),
            'max': np.max(sentence_bleus)
        }
        
        return results
    
    def calculate_chrf_detailed(self, predictions: List[str], references: List[List[str]]) -> Dict:
        """Calculate detailed chrF scores"""
        if not self.chrf_metric:
            return {'error': 'chrF metric not available'}
        
        results = {}
        
        # Flatten references for chrF calculation
        flat_references = [ref[0] if isinstance(ref, list) else ref for ref in references]
        
        # Corpus-level chrF
        try:
            corpus_chrf = self.chrf_metric.compute(
                predictions=predictions,
                references=flat_references
            )
            results['corpus_chrf'] = corpus_chrf['score']
        except Exception as e:
            results['corpus_chrf_error'] = str(e)
        
        # Sentence-level chrF scores
        sentence_chrfs = []
        for pred, ref in zip(predictions, flat_references):
            try:
                sent_chrf = self.chrf_metric.compute(
                    predictions=[pred],
                    references=[ref]
                )
                sentence_chrfs.append(sent_chrf['score'])
            except:
                sentence_chrfs.append(0.0)
        
        results['sentence_chrf'] = {
            'scores': sentence_chrfs,
            'mean': np.mean(sentence_chrfs),
            'std': np.std(sentence_chrfs),
            'min': np.min(sentence_chrfs),
            'max': np.max(sentence_chrfs)
        }
        
        return results
    
    def calculate_rouge_scores(self, predictions: List[str], references: List[str]) -> Dict:
        """Calculate ROUGE scores for additional evaluation"""
        if not self.rouge_metric:
            return {'error': 'ROUGE metric not available'}
        
        try:
            rouge_scores = self.rouge_metric.compute(
                predictions=predictions,
                references=references
            )
            return rouge_scores
        except Exception as e:
            return {'error': str(e)}
    
    def calculate_length_metrics(self, predictions: List[str], references: List[str]) -> Dict:
        """Calculate length-based metrics"""
        pred_lengths = [len(pred.split()) for pred in predictions]
        ref_lengths = [len(ref.split()) for ref in references]
        
        return {
            'avg_pred_length': np.mean(pred_lengths),
            'avg_ref_length': np.mean(ref_lengths),
            'length_ratio': np.mean(pred_lengths) / np.mean(ref_lengths),
            'length_diff_abs': [abs(p - r) for p, r in zip(pred_lengths, ref_lengths)],
            'length_diff_relative': [(p - r) / r for p, r in zip(pred_lengths, ref_lengths) if r > 0]
        }
    
    def evaluate_by_language(self, 
                           predictions: Dict[str, List[str]], 
                           references: Dict[str, List[str]]) -> Dict:
        """Evaluate translations by language"""
        results = {}
        
        for lang in predictions.keys():
            if lang not in references:
                continue
                
            lang_preds = predictions[lang]
            lang_refs = [[ref] for ref in references[lang]]
            
            # Calculate metrics for this language
            bleu_results = self.calculate_bleu_detailed(lang_preds, lang_refs)
            chrf_results = self.calculate_chrf_detailed(lang_preds, lang_refs)
            length_metrics = self.calculate_length_metrics(lang_preds, references[lang])
            rouge_scores = self.calculate_rouge_scores(lang_preds, references[lang])
            
            results[lang] = {
                'bleu': bleu_results,
                'chrf': chrf_results,
                'length': length_metrics,
                'rouge': rouge_scores,
                'sample_count': len(lang_preds)
            }
        
        return results
    
    def create_evaluation_report(self, evaluation_results: Dict) -> str:
        """Create a comprehensive evaluation report"""
        report = []
        report.append("=" * 80)
        report.append("MULTILINGUAL TRANSLATION EVALUATION REPORT")
        report.append("=" * 80)
        
        overall_bleu_scores = []
        overall_chrf_scores = []
        
        for lang, metrics in evaluation_results.items():
            if isinstance(metrics, dict) and 'bleu' in metrics:
                report.append(f"\nLanguage: {lang.upper()}")
                report.append("-" * 40)
                
                # BLEU scores
                if 'corpus_bleu' in metrics['bleu']:
                    corpus_bleu = metrics['bleu']['corpus_bleu']['score']
                    overall_bleu_scores.append(corpus_bleu)
                    report.append(f"BLEU Score: {corpus_bleu:.2f}")
                
                if 'sentence_bleu' in metrics['bleu']:
                    sent_bleu = metrics['bleu']['sentence_bleu']
                    report.append(f"  Avg Sentence BLEU: {sent_bleu['mean']:.2f} (±{sent_bleu['std']:.2f})")
                    report.append(f"  Range: {sent_bleu['min']:.2f} - {sent_bleu['max']:.2f}")
                
                # chrF scores
                if 'corpus_chrf' in metrics['chrf']:
                    corpus_chrf = metrics['chrf']['corpus_chrf']
                    overall_chrf_scores.append(corpus_chrf)
                    report.append(f"chrF Score: {corpus_chrf:.2f}")
                
                if 'sentence_chrf' in metrics['chrf']:
                    sent_chrf = metrics['chrf']['sentence_chrf']
                    report.append(f"  Avg Sentence chrF: {sent_chrf['mean']:.2f} (±{sent_chrf['std']:.2f})")
                
                # Length metrics
                if 'length' in metrics:
                    length_ratio = metrics['length']['length_ratio']
                    report.append(f"Length Ratio: {length_ratio:.2f}")
                
                # Sample count
                report.append(f"Samples Evaluated: {metrics['sample_count']}")
        
        # Overall statistics
        if overall_bleu_scores and overall_chrf_scores:
            report.append("\n" + "=" * 40)
            report.append("OVERALL STATISTICS")
            report.append("=" * 40)
            report.append(f"Average BLEU across languages: {np.mean(overall_bleu_scores):.2f}")
            report.append(f"Average chrF across languages: {np.mean(overall_chrf_scores):.2f}")
            report.append(f"Languages evaluated: {len(overall_bleu_scores)}")
        
        return "\n".join(report)
    
    def save_detailed_results(self, results: Dict, filename: str = "evaluation_results.csv"):
        """Save detailed results to CSV"""
        rows = []
        
        for lang, metrics in results.items():
            if not isinstance(metrics, dict):
                continue
                
            base_row = {'language': lang}
            
            # Add BLEU metrics
            if 'bleu' in metrics and 'corpus_bleu' in metrics['bleu']:
                base_row['corpus_bleu'] = metrics['bleu']['corpus_bleu']['score']
                base_row['bleu_bp'] = metrics['bleu']['corpus_bleu']['bp']
                
            if 'bleu' in metrics and 'sentence_bleu' in metrics['bleu']:
                base_row['avg_sentence_bleu'] = metrics['bleu']['sentence_bleu']['mean']
                base_row['std_sentence_bleu'] = metrics['bleu']['sentence_bleu']['std']
            
            # Add chrF metrics
            if 'chrf' in metrics and 'corpus_chrf' in metrics['chrf']:
                base_row['corpus_chrf'] = metrics['chrf']['corpus_chrf']
                
            if 'chrf' in metrics and 'sentence_chrf' in metrics['chrf']:
                base_row['avg_sentence_chrf'] = metrics['chrf']['sentence_chrf']['mean']
                base_row['std_sentence_chrf'] = metrics['chrf']['sentence_chrf']['std']
            
            # Add length metrics
            if 'length' in metrics:
                base_row['length_ratio'] = metrics['length']['length_ratio']
                base_row['avg_pred_length'] = metrics['length']['avg_pred_length']
                base_row['avg_ref_length'] = metrics['length']['avg_ref_length']
            
            # Add sample count
            base_row['sample_count'] = metrics.get('sample_count', 0)
            
            rows.append(base_row)
        
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        print(f"Detailed results saved to {filename}")

def create_sample_evaluation_data():
    """Create sample data for evaluation testing"""
    # Sample predictions and references for testing
    sample_data = {
        'hi': {
            'predictions': [
                'नमस्कार, आप कैसे हैं?',
                'मेरा नाम राम है।',
                'आज का मौसम बहुत अच्छा है।'
            ],
            'references': [
                'नमस्ते, आप कैसे हैं?',
                'मेरा नाम राम है।',
                'आज मौसम बहुत सुंदर है।'
            ]
        },
        'ta': {
            'predictions': [
                'வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?',
                'என் பெயர் ராம்.',
                'இன்று வானிலை மிகவும் நல்லது.'
            ],
            'references': [
                'வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?',
                'என் பெயர் ராம்.',
                'இன்று வானிலை மிகவும் அழகு.'
            ]
        }
    }
    return sample_data

def main():
    """Test the evaluation metrics"""
    print("Testing Evaluation Metrics for Multilingual Translation")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = DetailedEvaluationMetrics()
    
    # Get sample data
    sample_data = create_sample_evaluation_data()
    
    # Prepare data for evaluation
    predictions = {}
    references = {}
    
    for lang, data in sample_data.items():
        predictions[lang] = data['predictions']
        references[lang] = data['references']
    
    # Run evaluation
    results = evaluator.evaluate_by_language(predictions, references)
    
    # Generate report
    report = evaluator.create_evaluation_report(results)
    print(report)
    
    # Save detailed results
    evaluator.save_detailed_results(results)
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()