"""
Vietnamese Fake News Detection - CLI Interface

Usage:
    python main.py --text "Your Vietnamese news text here"
    python main.py --file path/to/news.txt
    python main.py --interactive
"""

import argparse
import torch
from transformers import AutoTokenizer
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model import HybridModel
from src.features import TextStyleExtractor
from src.rag_utils import RAGSearch
import src.preprocessing as preprocessing


def load_model(checkpoint_path='checkpoints/phobert_hybrid_model.pt', device='cpu'):
    """Load trained model from checkpoint"""
    print(f"Loading model from {checkpoint_path}...")
    
    model = HybridModel(model_name="vinai/phobert-base-v2", style_dim=10)
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # Handle checkpoint format (may have 'model_state_dict' key)
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
        print(f"✓ Model loaded (Validation F1: {state_dict.get('val_f1', 'N/A')})")
    else:
        model.load_state_dict(state_dict)
        print("✓ Model loaded")
    
    model.to(device).eval()
    return model


def predict(text, model, tokenizer, rag_searcher, style_extractor, device='cpu', max_len=256):
    """Predict if text is FAKE or REAL"""
    
    # Step 1: RAG Retrieval
    evidence, bm25_score, mean_idf = rag_searcher(text)
    
    # Step 2: Extract style features
    style_vec = style_extractor.get_style_vector(text, mean_idf=mean_idf, bm25_score=bm25_score)
    style_tensor = torch.tensor([style_vec], dtype=torch.float32).to(device)
    
    # Step 3: Preprocess and tokenize as sentence pair
    # FIX: Use sentence-pair tokenization [CLS] text [SEP] evidence [SEP]
    # with evidence capped at 25% of token budget to prevent KB real-news
    # patterns from overwhelming the input text's style signals.
    clean_t = preprocessing.clean_text(text)
    clean_e = preprocessing.clean_text(evidence) if evidence else None
    
    if clean_e:
        evidence_limit = 150
        evid_ids = tokenizer.encode(clean_e, add_special_tokens=False)
        if len(evid_ids) > evidence_limit:
            evid_ids = evid_ids[:evidence_limit]
            clean_e = tokenizer.decode(evid_ids, skip_special_tokens=True)
    
    encoding = tokenizer(
        clean_t,
        text_pair=clean_e,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Step 4: Predict
    with torch.no_grad():
        logits = model(input_ids, attention_mask, style_tensor)
        probs = torch.softmax(logits, dim=-1)
        predicted_label = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, predicted_label].item()
    
    return {
        'label': 'FAKE' if predicted_label == 0 else 'REAL',
        'confidence': confidence,
        'probabilities': {
            'FAKE': probs[0, 0].item(),
            'REAL': probs[0, 1].item()
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Vietnamese Fake News Detection')
    parser.add_argument('--text', type=str, help='Vietnamese news text to analyze')
    parser.add_argument('--file', type=str, help='Path to text file containing news')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--model', type=str, default='checkpoints/phobert_hybrid_model.pt', 
                       help='Path to model checkpoint')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    
    args = parser.parse_args()
    
    # Setup device
    device = 'cuda' if (args.gpu and torch.cuda.is_available()) else 'cpu'
    print(f"Using device: {device}")
    
    # Load components
    print("\n[1/4] Loading model...")
    model = load_model(args.model, device)
    
    print("[2/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    
    print("[3/4] Initializing RAG searcher...")
    rag_searcher = RAGSearch(database_jsonl_paths='Organized', cache_path='rag_cache')
    
    print("[4/4] Loading style extractor...")
    style_extractor = TextStyleExtractor()
    
    print("\n✓ All components ready!\n")
    
    # Interactive mode
    if args.interactive:
        print("=" * 80)
        print("INTERACTIVE FAKE NEWS DETECTION")
        print("=" * 80)
        print("Enter Vietnamese news text (type 'quit' to exit)\n")
        
        while True:
            text = input("News text: ")
            if text.lower() in ['quit', 'exit', 'q']:
                break
            
            if not text.strip():
                continue
            
            result = predict(text, model, tokenizer, rag_searcher, style_extractor, device)
            
            print(f"\n{'='*80}")
            print(f"Prediction: {result['label']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Probabilities: REAL={result['probabilities']['REAL']:.2%}, "
                  f"FAKE={result['probabilities']['FAKE']:.2%}")
            print(f"{'='*80}\n")
    
    # File mode
    elif args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        print(f"Analyzing file: {args.file}\n")
        result = predict(text, model, tokenizer, rag_searcher, style_extractor, device)
        
        print(f"{'='*80}")
        print(f"Prediction: {result['label']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"{'='*80}")
    
    # Text mode
    elif args.text:
        result = predict(args.text, model, tokenizer, rag_searcher, style_extractor, device)
        
        print(f"{'='*80}")
        print(f"Prediction: {result['label']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"{'='*80}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
