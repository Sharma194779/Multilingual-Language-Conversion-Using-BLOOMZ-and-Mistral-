# Multilingual Language Conversion Using BLOOMZ and Mistral
# English to 22 Indian Languages Translation with LoRA Fine-tuning

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from datasets import Dataset, load_dataset
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import sacrebleu
from evaluate import load
import warnings
warnings.filterwarnings("ignore")

# Configuration for the translation system
class TranslationConfig:
    """Configuration class for multilingual translation system"""
    
    # Model configurations
    BLOOMZ_MODEL = "bigscience/bloomz-7b1"
    MISTRAL_MODEL = "mistralai/Mistral-7B-v0.1"
    
    # Indian Languages (22 scheduled languages)
    INDIAN_LANGUAGES = {
        'hi': 'Hindi', 'bn': 'Bengali', 'te': 'Telugu', 'mr': 'Marathi',
        'ta': 'Tamil', 'ur': 'Urdu', 'gu': 'Gujarati', 'kn': 'Kannada',
        'ml': 'Malayalam', 'or': 'Odia', 'pa': 'Punjabi', 'as': 'Assamese',
        'mai': 'Maithili', 'bh': 'Bhojpuri', 'sd': 'Sindhi', 'sa': 'Sanskrit',
        'ne': 'Nepali', 'ks': 'Kashmiri', 'ko': 'Konkani', 'doi': 'Dogri',
        'mni': 'Manipuri', 'sat': 'Santhali'
    }
    
    # LoRA configuration
    LORA_CONFIG = {
        'r': 16,
        'lora_alpha': 32,
        'lora_dropout': 0.1,
        'bias': 'none',
        'task_type': TaskType.CAUSAL_LM,
        'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    }
    
    # Training configuration
    TRAINING_CONFIG = {
        'output_dir': './translation_models',
        'num_train_epochs': 3,
        'per_device_train_batch_size': 4,
        'per_device_eval_batch_size': 4,
        'gradient_accumulation_steps': 4,
        'learning_rate': 2e-4,
        'warmup_steps': 100,
        'logging_steps': 50,
        'save_steps': 500,
        'evaluation_strategy': 'steps',
        'eval_steps': 500,
        'save_total_limit': 2,
        'load_best_model_at_end': True,
        'fp16': True,
        'dataloader_pin_memory': False,
        'remove_unused_columns': False
    }

class MultilingualTranslator:
    """Main translator class handling BLOOMZ and Mistral models"""
    
    def __init__(self, model_name: str = None):
        self.config = TranslationConfig()
        self.model_name = model_name or self.config.BLOOMZ_MODEL
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_model(self, load_in_8bit: bool = True):
        """Load the base model and tokenizer"""
        print(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=load_in_8bit,
            trust_remote_code=True
        )
        
        print(f"Model loaded successfully on {self.device}")
        
    def setup_lora(self):
        """Setup LoRA configuration for parameter-efficient fine-tuning"""
        lora_config = LoraConfig(
            r=self.config.LORA_CONFIG['r'],
            lora_alpha=self.config.LORA_CONFIG['lora_alpha'],
            target_modules=self.config.LORA_CONFIG['target_modules'],
            lora_dropout=self.config.LORA_CONFIG['lora_dropout'],
            bias=self.config.LORA_CONFIG['bias'],
            task_type=self.config.LORA_CONFIG['task_type']
        )
        
        self.peft_model = get_peft_model(self.model, lora_config)
        self.peft_model.print_trainable_parameters()
        
        return lora_config
    
    def create_prompts(self, text: str, target_lang: str, source_lang: str = 'en') -> str:
        """Create appropriate prompts for translation"""
        source_name = 'English' if source_lang == 'en' else self.config.INDIAN_LANGUAGES.get(source_lang, source_lang)
        target_name = self.config.INDIAN_LANGUAGES.get(target_lang, target_lang)
        
        if 'bloomz' in self.model_name.lower():
            prompt = f"Translate from {source_name} to {target_name}: {text}\nTranslation:"
        else:  # Mistral
            prompt = f"<s>[INST] Translate the following {source_name} text to {target_name}: {text} [/INST]"
        
        return prompt
    
    def prepare_training_data(self, data_path: str = None) -> Dataset:
        """Prepare training data for fine-tuning"""
        if data_path:
            # Load custom dataset
            df = pd.read_csv(data_path)
        else:
            # Create synthetic training data (replace with actual parallel corpus)
            training_samples = []
            sample_texts = [
                "Hello, how are you?",
                "What is your name?",
                "I am learning a new language.",
                "The weather is beautiful today.",
                "Thank you very much.",
                "Where is the nearest hospital?",
                "I would like to order food.",
                "Can you help me?",
                "Good morning, have a nice day.",
                "This is a wonderful place to visit."
            ]
            
            # Note: In practice, you should use actual parallel corpus like:
            # - AI4Bharat's BPCC (Bharat Parallel Corpus Collection)
            # - IndicTrans2 dataset
            # - FLORES-200 for Indian languages
            
            for text in sample_texts:
                for lang_code in list(self.config.INDIAN_LANGUAGES.keys())[:5]:  # Use first 5 languages for demo
                    prompt = self.create_prompts(text, lang_code)
                    # Placeholder translation (replace with actual translations)
                    target = f"[{lang_code}] Translation of: {text}"
                    training_samples.append({
                        'input_text': prompt,
                        'target_text': target,
                        'source_lang': 'en',
                        'target_lang': lang_code
                    })
            
            df = pd.DataFrame(training_samples)
        
        # Convert to Hugging Face dataset
        dataset = Dataset.from_pandas(df)
        
        def tokenize_function(examples):
            # Combine input and target for causal language modeling
            full_texts = [f"{inp} {tgt}" for inp, tgt in zip(examples['input_text'], examples['target_text'])]
            
            tokenized = self.tokenizer(
                full_texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
            
            tokenized['labels'] = tokenized['input_ids'].clone()
            return tokenized
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Split into train/validation
        train_size = int(0.9 * len(tokenized_dataset))
        train_dataset = tokenized_dataset.select(range(train_size))
        eval_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))
        
        return train_dataset, eval_dataset
    
    def fine_tune(self, train_dataset: Dataset, eval_dataset: Dataset):
        """Fine-tune the model using LoRA"""
        print("Starting fine-tuning with LoRA...")
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Training arguments
        training_args = TrainingArguments(**self.config.TRAINING_CONFIG)
        
        # Trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        # Train
        trainer.train()
        
        # Save the model
        trainer.save_model()
        print("Fine-tuning completed!")
        
        return trainer
    
    def translate(self, text: str, target_lang: str, max_length: int = 100) -> str:
        """Translate text to target language"""
        prompt = self.create_prompts(text, target_lang)
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=400
        ).to(self.device)
        
        # Generate translation
        model_to_use = self.peft_model if self.peft_model else self.model
        
        with torch.no_grad():
            outputs = model_to_use.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract translation (remove prompt)
        translation = generated_text.replace(prompt, "").strip()
        
        return translation

class EvaluationMetrics:
    """Class for evaluating translation quality using BLEU and chrF"""
    
    def __init__(self):
        self.chrf = load("chrf")
        
    def calculate_bleu(self, predictions: List[str], references: List[List[str]]) -> float:
        """Calculate BLEU score"""
        bleu = sacrebleu.corpus_bleu(predictions, list(zip(*references)))
        return bleu.score
    
    def calculate_chrf(self, predictions: List[str], references: List[List[str]]) -> float:
        """Calculate chrF score"""
        # Flatten references for chrF calculation
        flat_references = [ref[0] for ref in references]
        
        chrf_score = self.chrf.compute(
            predictions=predictions,
            references=flat_references
        )
        return chrf_score['score']
    
    def evaluate_model(self, translator: MultilingualTranslator, test_data: List[Dict]) -> Dict:
        """Comprehensive evaluation of the translation model"""
        results = {}
        
        for lang in translator.config.INDIAN_LANGUAGES.keys():
            lang_predictions = []
            lang_references = []
            
            for sample in test_data:
                if sample['target_lang'] == lang:
                    prediction = translator.translate(sample['source_text'], lang)
                    lang_predictions.append(prediction)
                    lang_references.append([sample['reference']])
            
            if lang_predictions:
                bleu_score = self.calculate_bleu(lang_predictions, lang_references)
                chrf_score = self.calculate_chrf(lang_predictions, lang_references)
                
                results[lang] = {
                    'bleu': bleu_score,
                    'chrf': chrf_score,
                    'samples': len(lang_predictions)
                }
        
        return results

def create_sample_test_data() -> List[Dict]:
    """Create sample test data for evaluation"""
    test_samples = [
        {
            'source_text': 'Good morning',
            'target_lang': 'hi',
            'reference': 'सुप्रभात'
        },
        {
            'source_text': 'Thank you',
            'target_lang': 'ta',
            'reference': 'நன்றி'
        },
        {
            'source_text': 'How are you?',
            'target_lang': 'bn',
            'reference': 'আপনি কেমন আছেন?'
        }
    ]
    return test_samples

def main():
    """Main execution function"""
    print("Multilingual Language Conversion Using BLOOMZ and Mistral")
    print("=" * 60)
    
    # Initialize translator with BLOOMZ
    print("\n1. Setting up BLOOMZ translator...")
    bloomz_translator = MultilingualTranslator(TranslationConfig.BLOOMZ_MODEL)
    bloomz_translator.load_model()
    bloomz_translator.setup_lora()
    
    # Prepare training data
    print("\n2. Preparing training data...")
    train_data, eval_data = bloomz_translator.prepare_training_data()
    
    # Fine-tune the model
    print("\n3. Fine-tuning with LoRA...")
    trainer = bloomz_translator.fine_tune(train_data, eval_data)
    
    # Test translations
    print("\n4. Testing translations...")
    test_texts = [
        "Hello, how are you today?",
        "I love learning new languages.",
        "The weather is very nice."
    ]
    
    for text in test_texts:
        print(f"\nSource: {text}")
        for lang_code in ['hi', 'ta', 'bn']:
            translation = bloomz_translator.translate(text, lang_code)
            lang_name = TranslationConfig.INDIAN_LANGUAGES[lang_code]
            print(f"{lang_name} ({lang_code}): {translation}")
    
    # Evaluation
    print("\n5. Evaluating model performance...")
    evaluator = EvaluationMetrics()
    test_data = create_sample_test_data()
    results = evaluator.evaluate_model(bloomz_translator, test_data)
    
    print("\nEvaluation Results:")
    for lang, metrics in results.items():
        lang_name = TranslationConfig.INDIAN_LANGUAGES[lang]
        print(f"{lang_name} ({lang}):")
        print(f"  BLEU: {metrics['bleu']:.2f}")
        print(f"  chrF: {metrics['chrf']:.2f}")
        print(f"  Samples: {metrics['samples']}")
    
    print("\n" + "=" * 60)
    print("Translation system setup complete!")
    print("\nTo use with Mistral model, change the model_name parameter:")
    print("mistral_translator = MultilingualTranslator(TranslationConfig.MISTRAL_MODEL)")

if __name__ == "__main__":
    main()