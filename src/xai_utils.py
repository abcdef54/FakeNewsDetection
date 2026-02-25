# xai_utils.py
"""
Explainable AI Utilities for Fake News Detection.

This module provides tools to understand why the model makes specific predictions
using Integrated Gradients attribution method from the Captum library.
"""

import torch
import numpy as np
from captum.attr import LayerIntegratedGradients
from typing import List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


def explain_prediction(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                       style_vector: torch.Tensor, target_class: int = None) -> np.ndarray:
    """
    Computes the contribution of each input token to the final prediction.
    
    Uses Integrated Gradients to compute attribution scores for each token,
    showing which words influenced the model's decision.
    
    Args:
        model: Trained HybridModel instance
        input_ids: Token IDs [1, seq_len] or [seq_len]
        attention_mask: Attention mask [1, seq_len] or [seq_len]
        style_vector: Style features [1, 10] or [10]
        target_class: Class to explain (0=Real, 1=Fake). If None, uses predicted class.
    
    Returns:
        importances: Attribution scores for each token [seq_len]
    """
    model.eval()
    model.zero_grad()
    
    # Ensure batch dimension
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)
    if style_vector.dim() == 1:
        style_vector = style_vector.unsqueeze(0)
    
    # Move to model's device
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    style_vector = style_vector.to(device)
    
    # If no target class specified, use the model's prediction
    if target_class is None:
        with torch.no_grad():
            logits = model(input_ids, attention_mask, style_vector)
            target_class = torch.argmax(logits, dim=-1).item()
    
    # 1. Define Forward Function for Captum
    # Captum needs a function that takes the layer input and returns logits
    def forward_func(inputs, mask, style):
        logits = model(inputs, mask, style)
        return logits
    
    # 2. Initialize Integrated Gradients
    # Target the embedding layer to see which input tokens matter
    lig = LayerIntegratedGradients(forward_func, model.bert.embeddings.word_embeddings)
    
    # 3. Compute Attributions
    # This calculates the importance of each token for the target class
    attributions, delta = lig.attribute(
        inputs=input_ids,
        baselines=input_ids * 0,  # Baseline is all zeros (padding token)
        additional_forward_args=(attention_mask, style_vector),
        target=target_class,
        return_convergence_delta=True,
        n_steps=50  # Number of interpolation steps
    )
    
    # 4. Aggregate scores across embedding dimension
    # attributions shape: [1, seq_len, embedding_dim] -> [seq_len]
    importances = attributions.sum(dim=-1).squeeze(0)
    
    # 5. Normalize scores
    importances = importances / (torch.norm(importances) + 1e-10)
    
    return importances.detach().cpu().numpy()


def visualize_importance(tokenizer, input_ids: torch.Tensor, importances: np.ndarray, 
                        top_k: int = 20, save_path: str = None) -> str:
    """
    Visualizes token importance scores with color-coded HTML output.
    
    Args:
        tokenizer: HuggingFace tokenizer used to decode tokens
        input_ids: Token IDs [seq_len] or [1, seq_len]
        importances: Attribution scores [seq_len]
        top_k: Number of top important tokens to highlight
        save_path: Optional path to save visualization as HTML
    
    Returns:
        html: HTML string with color-coded words
    """
    # Ensure 1D
    if isinstance(input_ids, torch.Tensor):
        if input_ids.dim() == 2:
            input_ids = input_ids.squeeze(0)
        input_ids = input_ids.cpu().numpy()
    
    # Decode tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    # Normalize importances to [0, 1] for color mapping
    importances = np.array(importances)
    # Use absolute values for coloring (both positive and negative are important)
    abs_importances = np.abs(importances)
    
    if abs_importances.max() > 0:
        normalized_scores = abs_importances / abs_importances.max()
    else:
        normalized_scores = abs_importances
    
    # Build HTML output
    html_parts = ['<div style="line-height: 2.0; font-size: 14px;">']
    
    for token, score, raw_score in zip(tokens, normalized_scores, importances):
        # Skip special tokens
        if token in ['<s>', '</s>', '<pad>', '[CLS]', '[SEP]', '[PAD]']:
            continue
            
        # Clean PhoBERT word pieces (remove @@ symbols)
        display_token = token.replace('@@', '')
        
        # Color based on importance
        # Green for positive attribution, Red for negative
        if raw_score > 0:
            # Positive contribution (supports the prediction)
            alpha = min(score * 2, 1.0)  # Scale for visibility
            color = f'rgba(0, 255, 0, {alpha})'
        else:
            # Negative contribution (against the prediction)
            alpha = min(score * 2, 1.0)
            color = f'rgba(255, 0, 0, {alpha})'
        
        html_parts.append(
            f'<span style="background-color: {color}; padding: 2px 4px; '
            f'margin: 2px; border-radius: 3px; display: inline-block;">'
            f'{display_token}</span>'
        )
    
    html_parts.append('</div>')
    html = ''.join(html_parts)
    
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(f'<html><head><meta charset="utf-8"></head><body>{html}</body></html>')
    
    return html


def plot_top_attributions(tokenizer, input_ids: torch.Tensor, importances: np.ndarray, 
                          top_k: int = 15, save_path: str = None):
    """
    Creates a bar plot of the top-k most important tokens.
    
    Args:
        tokenizer: HuggingFace tokenizer
        input_ids: Token IDs [seq_len] or [1, seq_len]
        importances: Attribution scores [seq_len]
        top_k: Number of top tokens to display
        save_path: Optional path to save the plot
    """
    # Ensure 1D
    if isinstance(input_ids, torch.Tensor):
        if input_ids.dim() == 2:
            input_ids = input_ids.squeeze(0)
        input_ids = input_ids.cpu().numpy()
    
    # Decode tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    # Filter out special tokens
    filtered_pairs = []
    for token, score in zip(tokens, importances):
        if token not in ['<s>', '</s>', '<pad>', '[CLS]', '[SEP]', '[PAD]']:
            filtered_pairs.append((token.replace('@@', ''), score))
    
    # Sort by absolute importance
    sorted_pairs = sorted(filtered_pairs, key=lambda x: abs(x[1]), reverse=True)[:top_k]
    sorted_pairs = sorted_pairs[::-1]  # Reverse for better plot layout
    
    # Extract tokens and scores
    top_tokens = [pair[0] for pair in sorted_pairs]
    top_scores = [pair[1] for pair in sorted_pairs]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['green' if score > 0 else 'red' for score in top_scores]
    
    ax.barh(range(len(top_tokens)), top_scores, color=colors, alpha=0.7)
    ax.set_yticks(range(len(top_tokens)))
    ax.set_yticklabels(top_tokens)
    ax.set_xlabel('Attribution Score')
    ax.set_title(f'Top {top_k} Most Important Tokens')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def get_style_feature_importance(model, input_ids: torch.Tensor, 
                                 attention_mask: torch.Tensor, 
                                 style_vector: torch.Tensor,
                                 feature_names: List[str] = None) -> dict:
    """
    Analyzes the importance of each style feature using gradient-based attribution.
    
    Args:
        model: Trained HybridModel
        input_ids: Token IDs
        attention_mask: Attention mask
        style_vector: Style features [1, 10] or [10]
        feature_names: Names of the 10 style features
        
    Returns:
        dict: Feature names mapped to importance scores
    """
    if feature_names is None:
        feature_names = [
            'sentiment_score', 'subjectivity_score', 'word_count_normalized',
            'lexical_diversity', 'mean_idf', 'caps_ratio', 'punct_ratio',
            'pronoun_density', 'typo_density', 'max_bm25_score'
        ]
    
    model.eval()
    
    # Ensure proper dimensions
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)
    if style_vector.dim() == 1:
        style_vector = style_vector.unsqueeze(0)
    
    # Move to device and enable gradient
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    style_vector = style_vector.to(device).requires_grad_(True)
    
    # Forward pass
    logits = model(input_ids, attention_mask, style_vector)
    predicted_class = torch.argmax(logits, dim=-1)
    
    # Backward pass to get gradients
    model.zero_grad()
    logits[0, predicted_class].backward()
    
    # Get gradients of style features
    style_grads = style_vector.grad.squeeze(0).abs().cpu().numpy()
    
    # Create importance dictionary
    importance_dict = {name: float(score) for name, score in zip(feature_names, style_grads)}
    
    return importance_dict

    """
    Prints words colored by their importance score.
    """
    
    print("\n--- XAI Explanation (Why did the model choose this?) ---")
    