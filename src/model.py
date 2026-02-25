# model.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Tuple, Optional
import os

# Default configuration values
STYLE_VECTOR_DIM = 10

class HybridModel(nn.Module):
    """
    Hybrid Model Architecture for Fake News Detection.
    
    This model combines:
    1. BERT-based language understanding (PhoBERT or ViSoBERT)
    2. Handcrafted stylistic features (10-dimensional style vector)
    
    Architecture:
        BERT Encoder (768-dim) + Style Vector (10-dim) = 778-dim fusion
        → MLP Classifier → 2 classes (Fake/Real)
    
    Args:
        model_name (str): HuggingFace model name (e.g., "vinai/phobert-base", "uitnlp/visobert")
        style_dim (int): Dimension of style vector (default: 10)
        num_labels (int): Number of output classes (default: 2)
        hidden_dim (int): Hidden layer dimension (default: 256)
        dropout (float): Dropout rate (default: 0.3)
    """
    def __init__(self, model_name: str, style_dim: int = STYLE_VECTOR_DIM, 
                 num_labels: int = 2, hidden_dim: int = 256, dropout: float = 0.3):
        super(HybridModel, self).__init__()
        
        # 1. BERT Encoder
        self.bert = AutoModel.from_pretrained(model_name)
        
        # 2. Fusion Layer Configuration
        # Input = BERT_CLS (768) + Style (10) = 778
        bert_dim = self.bert.config.hidden_size  # 768 for most Vietnamese BERT models
        self.style_dim = style_dim
        self.fusion_dim = bert_dim + style_dim  # 778 total
        
        # 3. Classifier Head (MLP)
        # 778 → hidden_dim → num_labels
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_labels)
        )
        
        print(f"[INFO] HybridModel initialized with {model_name}")
        print(f"[INFO] Fusion dimension: {bert_dim} (BERT) + {style_dim} (Style) = {self.fusion_dim}")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                style_vector: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the hybrid model.
        
        Args:
            input_ids: Token IDs [Batch, Max_Len]
            attention_mask: Attention mask [Batch, Max_Len]
            style_vector: Style features [Batch, 10]
            
        Returns:
            logits: Class logits [Batch, 2]
        """
        # A. BERT Forward Pass
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [Batch, 768]
        
        # B. Concatenate BERT embedding with style vector
        # Cast style_vector to match BERT output dtype (handles FP16 inference)
        style_vector = style_vector.to(dtype=pooled_output.dtype)
        combined = torch.cat((pooled_output, style_vector), dim=1)  # [Batch, 778]
        
        # C. Classify
        logits = self.classifier(combined)  # [Batch, 2]
        return logits
    

class EnsembleWrapper:
    """
    Ensemble Inference Wrapper with Soft Voting.
    
    This class loads two independently trained HybridModel instances:
    - Model A: PhoBERT expert (better for formal news articles)
    - Model B: ViSoBERT expert (better for social media text)
    
    Each model uses its OWN tokenizer — PhoBERT and ViSoBERT have
    completely different vocabularies, so the same text must be tokenized
    separately for each model.
    
    Prediction Strategy:
        Soft Voting — Average the probability distributions from both models
        and select the class with the highest averaged probability.
    
    Label Encoding:
        0 = FAKE,  1 = REAL
    
    Args:
        phobert_path (str): Path to trained PhoBERT checkpoint (.pth)
        visobert_path (str): Path to trained ViSoBERT checkpoint (.pth)
        device (str): Device to run inference on ('cuda' or 'cpu')
        max_len (int): Maximum token length (must match training, default 256)
    """
    def __init__(self, phobert_path: str, visobert_path: str,
                 device: str = 'cpu', max_len: int = 256):
        self.device = torch.device(device)
        self.max_len = max_len
        
        print(f"[INFO] Loading Ensemble Models on {self.device}...")
        
        # 1. Load Tokenizers (each model needs its own)
        self.tokenizer_a = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
        self.tokenizer_b = AutoTokenizer.from_pretrained("uitnlp/visobert", use_fast=False)
        
        # 2. Initialize Model Architectures
        # Expert A: PhoBERT (Formal News)
        self.model_a = HybridModel(model_name="vinai/phobert-base-v2", style_dim=STYLE_VECTOR_DIM)
        
        # Expert B: ViSoBERT (Social Media)
        self.model_b = HybridModel(model_name="uitnlp/visobert", style_dim=STYLE_VECTOR_DIM)
        
        # FIX: Resize ViSoBERT embeddings to match its tokenizer vocab
        self.model_b.bert.resize_token_embeddings(len(self.tokenizer_b))
        
        # 3. Load Trained Weights (checkpoints save dict with 'model_state_dict' key)
        #    Load to CPU first, then move to device one-by-one to avoid OOM.
        if not os.path.exists(phobert_path):
            raise FileNotFoundError(f"PhoBERT model not found: {phobert_path}")
        if not os.path.exists(visobert_path):
            raise FileNotFoundError(f"ViSoBERT model not found: {visobert_path}")
        
        # Model A: load weights → move to device → eval
        checkpoint_a = torch.load(phobert_path, map_location='cpu')
        self.model_a.load_state_dict(checkpoint_a['model_state_dict'])
        del checkpoint_a
        self.model_a.to(self.device).eval()
        
        # Model B: load weights → move to device → eval
        checkpoint_b = torch.load(visobert_path, map_location='cpu')
        self.model_b.load_state_dict(checkpoint_b['model_state_dict'])
        del checkpoint_b
        self.model_b.to(self.device).eval()
        
        print("[INFO] Ensemble Ready (dual-tokenizer mode).")

    # ── helper ──────────────────────────────────────────────
    def _tokenize(self, text: str, tokenizer, evidence: Optional[str] = None):
        """Tokenize text (optionally with evidence as sentence pair) and return tensors.
        
        When evidence is provided, it is encoded as the second segment of a
        sentence pair: [CLS] text [SEP] evidence [SEP].  Evidence tokens are
        hard-capped at 150 so the input text's style signals
        (caps, emotion words, punctuation) remain visible in the token sequence.
        """
        if evidence:
            # Pre-truncate evidence tokens to 150 (same as training)
            evidence_limit = 150
            evid_ids = tokenizer.encode(evidence, add_special_tokens=False)
            if len(evid_ids) > evidence_limit:
                evid_ids = evid_ids[:evidence_limit]
                evidence = tokenizer.decode(evid_ids, skip_special_tokens=True)
            enc = tokenizer(
                text,
                text_pair=evidence,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        else:
            enc = tokenizer(
                text,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        return enc['input_ids'].to(self.device), enc['attention_mask'].to(self.device)

    @staticmethod
    def _ensure_batch(*tensors):
        """Add batch dimension (dim-0) if missing."""
        out = []
        for t in tensors:
            out.append(t.unsqueeze(0) if t.dim() == 1 else t)
        return out

    # ── core call (pre-tokenized) ───────────────────────────
    @torch.no_grad()
    def __call__(
        self,
        input_ids_a: torch.Tensor, attention_mask_a: torch.Tensor,
        input_ids_b: torch.Tensor, attention_mask_b: torch.Tensor,
        style_vector: torch.Tensor,
    ) -> Tuple[int, float]:
        """
        Ensemble prediction from **pre-tokenized** inputs.
        
        Each model receives its OWN token IDs (different vocabularies).
        
        Args:
            input_ids_a / attention_mask_a: PhoBERT-tokenized  [B, L] or [L]
            input_ids_b / attention_mask_b: ViSoBERT-tokenized [B, L] or [L]
            style_vector: Style features [B, 10] or [10]
            
        Returns:
            predicted_label (int): 0 = FAKE, 1 = REAL
            confidence_score (float): Probability of the predicted class
        """
        input_ids_a, attention_mask_a, input_ids_b, attention_mask_b, style_vector = \
            self._ensure_batch(input_ids_a, attention_mask_a,
                               input_ids_b, attention_mask_b, style_vector)
        
        input_ids_a = input_ids_a.to(self.device)
        attention_mask_a = attention_mask_a.to(self.device)
        input_ids_b = input_ids_b.to(self.device)
        attention_mask_b = attention_mask_b.to(self.device)
        style_vector = style_vector.to(self.device)
        
        # A. Forward through each expert with its own tokens
        logits_a = self.model_a(input_ids_a, attention_mask_a, style_vector)  # [B, 2]
        logits_b = self.model_b(input_ids_b, attention_mask_b, style_vector)  # [B, 2]
        
        # B. Soft Voting
        probs_a = torch.softmax(logits_a, dim=-1)
        probs_b = torch.softmax(logits_b, dim=-1)
        avg_probs = (probs_a + probs_b) / 2.0
        
        predicted_label = torch.argmax(avg_probs, dim=-1).item()
        confidence_score = avg_probs[0, predicted_label].item()
        
        return predicted_label, confidence_score

    # ── convenience: text → prediction ──────────────────────
    @torch.no_grad()
    def predict_text(self, text: str, style_vector: torch.Tensor,
                     evidence: Optional[str] = None) -> Tuple[int, float]:
        """
        End-to-end prediction from raw text (+ optional evidence).
        
        Tokenizes the text with BOTH tokenizers internally.  When *evidence*
        is supplied it is encoded as the second segment of a sentence pair,
        matching the training format, with evidence tokens capped at 25 %
        of max_len.
        
        Args:
            text (str): Pre-processed Vietnamese text (already cleaned)
            style_vector: Style features tensor [10] or [1, 10]
            evidence (str, optional): Cleaned RAG evidence text
            
        Returns:
            predicted_label (int): 0 = FAKE, 1 = REAL
            confidence_score (float): Probability of the predicted class
        """
        ids_a, mask_a = self._tokenize(text, self.tokenizer_a, evidence=evidence)
        ids_b, mask_b = self._tokenize(text, self.tokenizer_b, evidence=evidence)
        
        if style_vector.dim() == 1:
            style_vector = style_vector.unsqueeze(0)
        
        return self(ids_a, mask_a, ids_b, mask_b, style_vector.to(self.device))

    # ── individual model analysis ───────────────────────────
    def get_individual_predictions(self, text: str, style_vector: torch.Tensor,
                                   evidence: Optional[str] = None) -> dict:
        """
        Get predictions from both models individually (for analysis).
        
        Args:
            text (str): Pre-processed Vietnamese text
            style_vector: Style features tensor [10] or [1, 10]
            evidence (str, optional): Cleaned RAG evidence text
        
        Returns:
            dict with 'phobert' and 'visobert' sub-dicts containing
            prediction, confidence, and full probability arrays.
        """
        ids_a, mask_a = self._tokenize(text, self.tokenizer_a, evidence=evidence)
        ids_b, mask_b = self._tokenize(text, self.tokenizer_b, evidence=evidence)
        
        if style_vector.dim() == 1:
            style_vector = style_vector.unsqueeze(0)
        style_vector = style_vector.to(self.device)
        
        with torch.no_grad():
            logits_a = self.model_a(ids_a, mask_a, style_vector)
            logits_b = self.model_b(ids_b, mask_b, style_vector)
            probs_a = torch.softmax(logits_a, dim=-1)
            probs_b = torch.softmax(logits_b, dim=-1)
        
        return {
            'phobert': {
                'prediction': torch.argmax(probs_a, dim=-1).item(),
                'confidence': torch.max(probs_a, dim=-1).values.item(),
                'probabilities': probs_a[0].cpu().numpy()
            },
            'visobert': {
                'prediction': torch.argmax(probs_b, dim=-1).item(),
                'confidence': torch.max(probs_b, dim=-1).values.item(),
                'probabilities': probs_b[0].cpu().numpy()
            }
        }
