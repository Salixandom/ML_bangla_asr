"""
Training Script for Bangla ASR Pipeline

Model: wav2vec-BERT 2.0 with CTC loss
Training features:
- Mixed precision (FP16)
- Gradient accumulation
- Learning rate warmup
- Feature encoder freezing
- Logging and checkpointing
"""

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.amp import GradScaler, autocast
from pathlib import Path
from typing import Dict, Optional, Tuple
from tqdm import tqdm
import numpy as np
from dataclasses import dataclass
import wandb
import json

from transformers import (
    Wav2Vec2BertForCTC,
    get_scheduler
)

from config import PipelineConfig, get_config, MODEL_PRESETS
from dataset import (
    BanglaVocabulary, 
    create_dataloaders
)


class PretrainedVocabularyWrapper:
    """
    Wrapper around a pretrained Wav2Vec2BertProcessor's tokenizer.
    
    Provides the same interface as BanglaVocabulary but uses the
    pretrained model's vocabulary, preserving CTC head compatibility.
    """
    
    def __init__(self, processor):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        
        # Get special token IDs
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        self.unk_token_id = self.tokenizer.unk_token_id if self.tokenizer.unk_token_id is not None else 1
        self.blank_token_id = self.pad_token_id  # CTC blank is usually pad
        self.word_delimiter_id = self.tokenizer.word_delimiter_token_id if hasattr(self.tokenizer, 'word_delimiter_token_id') else self.pad_token_id
        
        # Build vocab mappings
        self.vocab = self.tokenizer.get_vocab()
        self.idx_to_char = {v: k for k, v in self.vocab.items()}
    
    def __len__(self):
        return len(self.vocab)
    
    def encode(self, text: str) -> list:
        """Encode text to token IDs using the pretrained tokenizer."""
        # Use the processor's tokenizer
        encoded = self.tokenizer(text, return_tensors=None)
        return encoded['input_ids']
    
    def decode(self, token_ids: list, skip_special: bool = True) -> str:
        """Decode token IDs back to text."""
        # Filter out special tokens if requested
        if skip_special:
            special_ids = {self.pad_token_id, self.blank_token_id}
            token_ids = [t for t in token_ids if t not in special_ids]
        
        # Use the tokenizer's decode
        return self.tokenizer.decode(token_ids)
    
    def get_vocab_dict(self) -> dict:
        """Return vocabulary dictionary."""
        return self.vocab.copy()


class Wav2VecBertCTCModel(nn.Module):
    """
    wav2vec-BERT 2.0 model wrapper for CTC-based ASR.
    
    Supports:
    - Loading pretrained Bangla model with its vocabulary
    - Or custom vocabulary (reinitializes CTC head)
    """
    
    def __init__(
        self,
        config: PipelineConfig,
        vocab_size: int = None,
        pretrained_name: str = "facebook/w2v-bert-2.0",
        use_pretrained_vocab: bool = False
    ):
        super().__init__()
        
        self.config = config
        self.use_pretrained_vocab = use_pretrained_vocab
        
        if use_pretrained_vocab:
            # Load model with its original vocabulary (no CTC head reinitialization)
            self.model = Wav2Vec2BertForCTC.from_pretrained(
                pretrained_name,
                ctc_loss_reduction="mean",
                ctc_zero_infinity=config.model.ctc_zero_infinity,
            )
            print(f"Loaded pretrained model with original vocabulary")
        else:
            # Load model with custom vocabulary (reinitializes CTC head)
            self.model = Wav2Vec2BertForCTC.from_pretrained(
                pretrained_name,
                vocab_size=vocab_size,
                ctc_loss_reduction="mean",
                ctc_zero_infinity=config.model.ctc_zero_infinity,
                pad_token_id=0,
                ignore_mismatched_sizes=True
            )
            print(f"Loaded model with custom vocabulary (size={vocab_size})")
        
        # Print model structure for debugging
        print(f"Model structure: {[name for name, _ in self.model.wav2vec2_bert.named_children()]}")
        
        # Optionally freeze feature encoder
        if config.model.freeze_feature_encoder:
            self._freeze_feature_encoder()
    
    def _freeze_feature_encoder(self):
        """
        Freeze the feature encoder layers.
        
        wav2vec-BERT 2.0 has a different architecture than wav2vec2.
        We freeze the feature_projection and optionally early encoder layers.
        """
        # Freeze feature projection
        if hasattr(self.model.wav2vec2_bert, 'feature_projection'):
            for param in self.model.wav2vec2_bert.feature_projection.parameters():
                param.requires_grad = False
            print("Feature projection frozen")
        
        # Freeze adapter if it exists and is not None
        adapter = getattr(self.model.wav2vec2_bert, 'adapter', None)
        if adapter is not None:
            for param in adapter.parameters():
                param.requires_grad = False
            print("Adapter frozen")
        
        # Optionally freeze early encoder layers (first half)
        if hasattr(self.model.wav2vec2_bert, 'encoder'):
            encoder = self.model.wav2vec2_bert.encoder
            if hasattr(encoder, 'layers'):
                num_layers = len(encoder.layers)
                freeze_layers = num_layers // 2  # Freeze first half
                for i, layer in enumerate(encoder.layers[:freeze_layers]):
                    for param in layer.parameters():
                        param.requires_grad = False
                print(f"Frozen first {freeze_layers}/{num_layers} encoder layers")
    
    def _unfreeze_feature_encoder(self):
        """Unfreeze all encoder parameters."""
        for param in self.model.wav2vec2_bert.parameters():
            param.requires_grad = True
        print("All encoder layers unfrozen")
    
    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for wav2vec-BERT 2.0.
        
        Args:
            input_features: Extracted features (batch, seq_len, 160)
                           Use SeamlessM4TFeatureExtractor to extract.
            attention_mask: Mask for padded positions
            labels: Token IDs for CTC loss (-100 for padding)
            
        Returns:
            Dictionary with loss, logits
        """
        outputs = self.model(
            input_features=input_features,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return {
            'loss': outputs.loss,
            'logits': outputs.logits,
        }
    
    def decode_greedy(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Greedy CTC decoding.
        
        Args:
            logits: Model output logits (batch, time, vocab)
            
        Returns:
            Predicted token IDs (batch, time)
        """
        return torch.argmax(logits, dim=-1)
    
    def save_pretrained(self, save_path: Path):
        """Save model to directory."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_path)
        print(f"Model saved to {save_path}")
    
    @classmethod
    def from_pretrained(cls, load_path: Path, config: PipelineConfig, vocab_size: int):
        """Load model from directory."""
        instance = cls(config, vocab_size)
        instance.model = Wav2Vec2BertForCTC.from_pretrained(load_path)
        return instance


class CTCDecoder:
    """
    CTC decoding utilities.
    
    Supports:
    - Greedy decoding
    - Beam search (optional)
    
    NOTE: Hugging Face CTC uses pad_token_id as the blank token.
    """
    
    def __init__(self, vocabulary: BanglaVocabulary, blank_id: int = None):
        self.vocabulary = vocabulary
        # Use vocabulary's pad_token_id as CTC blank (Hugging Face convention)
        self.blank_id = blank_id if blank_id is not None else vocabulary.pad_token_id
    
    def decode_greedy(self, logits: torch.Tensor) -> list:
        """
        Greedy CTC decoding with blank/repeat removal.
        
        Args:
            logits: (batch, time, vocab) or (time, vocab)
            
        Returns:
            List of decoded strings
        """
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)
        
        # Get predictions
        predictions = torch.argmax(logits, dim=-1)
        
        decoded_batch = []
        for pred in predictions:
            # Remove consecutive duplicates and blanks
            tokens = []
            prev_token = None
            
            for token in pred.tolist():
                if token != prev_token and token != self.blank_id:
                    tokens.append(token)
                prev_token = token
            
            # Decode to text
            text = self.vocabulary.decode(tokens)
            decoded_batch.append(text)
        
        return decoded_batch
    
    def decode_beam(
        self, 
        logits: torch.Tensor, 
        beam_width: int = 100
    ) -> list:
        """
        Beam search CTC decoding.
        
        Requires pyctcdecode library for advanced beam search.
        Falls back to greedy if not available.
        """
        try:
            from pyctcdecode import build_ctcdecoder
            
            # Build decoder (cached)
            if not hasattr(self, '_beam_decoder'):
                vocab_list = [
                    self.vocabulary.idx_to_char.get(i, '')
                    for i in range(len(self.vocabulary))
                ]
                self._beam_decoder = build_ctcdecoder(vocab_list)
            
            # Decode
            if logits.dim() == 2:
                logits = logits.unsqueeze(0)
            
            decoded_batch = []
            log_probs = torch.log_softmax(logits, dim=-1).cpu().numpy()
            
            for lp in log_probs:
                text = self._beam_decoder.decode(lp, beam_width=beam_width)
                decoded_batch.append(text)
            
            return decoded_batch
            
        except ImportError:
            print("pyctcdecode not installed, falling back to greedy")
            return self.decode_greedy(logits)


def compute_wer(predictions: list, references: list) -> float:
    """
    Compute Word Error Rate.
    
    WER = (S + D + I) / N
    where S=substitutions, D=deletions, I=insertions, N=reference words
    """
    import jiwer
    
    # Filter empty strings
    pred_ref_pairs = [
        (p, r) for p, r in zip(predictions, references)
        if r.strip()  # Skip empty references
    ]
    
    if not pred_ref_pairs:
        return 0.0
    
    predictions = [p for p, _ in pred_ref_pairs]
    references = [r for _, r in pred_ref_pairs]
    
    wer = jiwer.wer(references, predictions)
    return wer


def compute_cer(predictions: list, references: list) -> float:
    """
    Compute Character Error Rate.
    """
    import jiwer
    
    # Filter empty strings
    pred_ref_pairs = [
        (p, r) for p, r in zip(predictions, references)
        if r.strip()
    ]
    
    if not pred_ref_pairs:
        return 0.0
    
    predictions = [p for p, _ in pred_ref_pairs]
    references = [r for _, r in pred_ref_pairs]
    
    cer = jiwer.cer(references, predictions)
    return cer


class Trainer:
    """
    Training loop for wav2vec-BERT 2.0 ASR.
    """
    
    def __init__(
        self,
        model: Wav2VecBertCTCModel,
        config: PipelineConfig,
        train_loader,
        valid_loader,
        vocabulary: BanglaVocabulary,
        output_dir: Path,
        use_wandb: bool = False
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.vocabulary = vocabulary
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_wandb = use_wandb
        
        # Setup device
        self.device = torch.device(
            config.device if torch.cuda.is_available() else 'cpu'
        )
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.model.learning_rate,
            weight_decay=config.model.weight_decay
        )
        
        # Setup scheduler
        total_steps = len(train_loader) * config.model.num_train_epochs
        self.scheduler = get_scheduler(
            'linear',
            optimizer=self.optimizer,
            num_warmup_steps=config.model.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Setup mixed precision
        self.scaler = GradScaler('cuda') if config.model.fp16 else None
        
        # Setup decoder
        self.decoder = CTCDecoder(vocabulary)
        
        # Training state
        self.global_step = 0
        self.best_wer = float('inf')
        self.start_epoch = 1  # Will be updated if resuming from checkpoint
        
        # Gradient accumulation
        self.gradient_accumulation_steps = config.model.gradient_accumulation_steps
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device - pre-extracted features
            input_features = batch['input_features'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass with mixed precision
            if self.config.model.fp16:
                with autocast('cuda'):
                    outputs = self.model(
                        input_features=input_features,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs['loss'] / self.gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(
                    input_features=input_features,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs['loss'] / self.gradient_accumulation_steps
                loss.backward()
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            
            # Gradient accumulation step
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.model.fp16:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.model.max_grad_norm
                )
                
                # Optimizer step
                if self.config.model.fp16:
                    # Check if optimizer step was skipped due to inf/nan gradients
                    old_scale = self.scaler.get_scale()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    new_scale = self.scaler.get_scale()
                    
                    # Only step scheduler if optimizer actually stepped
                    if old_scale <= new_scale:
                        self.scheduler.step()
                else:
                    self.optimizer.step()
                    self.scheduler.step()
                
                self.optimizer.zero_grad()
                
                self.global_step += 1
                
                # Unfreeze feature encoder after warmup (0 = never unfreeze)
                if (self.config.model.freeze_feature_encoder and 
                    self.config.model.freeze_feature_encoder_steps > 0 and
                    self.global_step == self.config.model.freeze_feature_encoder_steps):
                    self.model._unfreeze_feature_encoder()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{total_loss / num_batches:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Logging
            if self.global_step % self.config.model.logging_steps == 0:
                if self.use_wandb:
                    wandb.log({
                        'train/loss': total_loss / num_batches,
                        'train/lr': self.scheduler.get_last_lr()[0],
                        'train/step': self.global_step
                    })
        
        return {'loss': total_loss / num_batches}
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()
        
        total_loss = 0
        all_predictions = []
        all_references = []
        num_batches = 0
        
        for batch in tqdm(self.valid_loader, desc="Evaluating"):
            input_features = batch['input_features'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            outputs = self.model(
                input_features=input_features,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs['loss'].item()
            num_batches += 1
            
            # Decode predictions
            predictions = self.decoder.decode_greedy(outputs['logits'])
            all_predictions.extend(predictions)
            
            # Decode references (remove padding)
            for label in labels:
                label_ids = label[label != -100].tolist()
                ref_text = self.vocabulary.decode(label_ids)
                all_references.append(ref_text)
        
        # Compute metrics
        wer = compute_wer(all_predictions, all_references)
        cer = compute_cer(all_predictions, all_references)
        
        metrics = {
            'loss': total_loss / num_batches,
            'wer': wer,
            'cer': cer
        }
        
        print(f"\nEvaluation: Loss={metrics['loss']:.4f}, WER={wer:.4f}, CER={cer:.4f}")
        
        # Save sample predictions to file (avoids terminal font issues)
        samples_file = self.output_dir / 'sample_predictions.txt'
        with open(samples_file, 'w', encoding='utf-8') as f:
            f.write(f"Step: {self.global_step}\n")
            f.write(f"WER: {wer:.4f}, CER: {cer:.4f}\n\n")
            for i in range(min(10, len(all_predictions))):
                f.write(f"[{i+1}] Ref:  {all_references[i]}\n")
                f.write(f"    Pred: {all_predictions[i]}\n\n")
        print(f"Sample predictions saved to {samples_file}")
        
        return metrics
    
    def save_checkpoint(self, name: str, keep_latest: int = None, epoch: int = None):
        """
        Save training checkpoint.
        
        Args:
            name: Checkpoint name (e.g., 'best', 'epoch_1', 'step_1000')
            keep_latest: If set, keep only the latest N epoch checkpoints
            epoch: Current epoch number to save
        """
        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(checkpoint_dir)
        
        # Save training state (including epoch and scaler!)
        state = {
            'global_step': self.global_step,
            'best_wer': self.best_wer,
            'epoch': epoch if epoch is not None else 0,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
        
        # Save scaler state for mixed precision
        if self.scaler is not None:
            state['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(state, checkpoint_dir / 'training_state.pt')
        
        # Save vocabulary
        self.vocabulary.save(checkpoint_dir / 'vocabulary.json')
        
        print(f"Model saved to {checkpoint_dir}")
        
        # Clean up old epoch checkpoints (keep only latest N)
        if keep_latest is not None and name.startswith('epoch_'):
            self._cleanup_old_checkpoints(keep_latest)
    
    def _cleanup_old_checkpoints(self, keep_latest: int = 3):
        """
        Remove old epoch checkpoints, keeping only the latest N.
        Never removes 'best' or 'step_*' checkpoints.
        """
        import re
        
        # Find all epoch checkpoints
        epoch_checkpoints = []
        for d in self.output_dir.iterdir():
            if d.is_dir() and d.name.startswith('epoch_'):
                match = re.match(r'epoch_(\d+)', d.name)
                if match:
                    epoch_num = int(match.group(1))
                    epoch_checkpoints.append((epoch_num, d))
        
        # Sort by epoch number (descending)
        epoch_checkpoints.sort(key=lambda x: x[0], reverse=True)
        
        # Remove old checkpoints
        for epoch_num, checkpoint_dir in epoch_checkpoints[keep_latest:]:
            print(f"Removing old checkpoint: {checkpoint_dir.name}")
            import shutil
            shutil.rmtree(checkpoint_dir)
    
    def load_checkpoint(self, checkpoint_dir: Path):
        """Load training checkpoint including model weights."""
        checkpoint_dir = Path(checkpoint_dir)
        
        # Load model weights from checkpoint
        print(f"Loading model weights from {checkpoint_dir}...")
        self.model.model = Wav2Vec2BertForCTC.from_pretrained(checkpoint_dir)
        self.model = self.model.to(self.device)
        
        # IMPORTANT: Recreate optimizer with NEW model parameters
        # (The old optimizer was pointing to the old model's parameters)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.model.learning_rate,
            weight_decay=self.config.model.weight_decay
        )
        
        # Load training state
        state = torch.load(checkpoint_dir / 'training_state.pt', map_location=self.device)
        self.global_step = state['global_step']
        self.best_wer = state['best_wer']
        
        # Load optimizer state
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        
        # Load scheduler state
        self.scheduler.load_state_dict(state['scheduler_state_dict'])
        
        # Load scaler state if available (for mixed precision)
        if 'scaler_state_dict' in state and self.scaler is not None:
            self.scaler.load_state_dict(state['scaler_state_dict'])
        
        # Recover epoch (with fallback for old checkpoints without epoch field)
        if 'epoch' in state and state['epoch'] > 0:
            self.start_epoch = state['epoch'] + 1  # Resume from next epoch
        else:
            # Estimate epoch from global_step
            steps_per_epoch = len(self.train_loader)
            estimated_epoch = self.global_step // steps_per_epoch
            self.start_epoch = max(estimated_epoch, 1)  # At least epoch 1
            print(f"  (Epoch estimated from global_step {self.global_step}: ~{estimated_epoch})")
        
        print(f"Checkpoint loaded from {checkpoint_dir}")
        print(f"  Resuming from epoch {self.start_epoch}, step {self.global_step}, best WER: {self.best_wer:.4f}")
    
    def train(self):
        """Full training loop."""
        total_epochs = self.config.model.num_train_epochs
        remaining_epochs = total_epochs - self.start_epoch + 1
        
        print(f"Starting training from epoch {self.start_epoch} to {total_epochs}")
        print(f"Remaining epochs: {remaining_epochs}")
        print(f"Device: {self.device}")
        
        for epoch in range(self.start_epoch, total_epochs + 1):
            train_metrics = self.train_epoch(epoch)
            
            # End of epoch evaluation
            eval_metrics = self.evaluate()
            
            print(f"\nEpoch {epoch} complete:")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Valid WER: {eval_metrics['wer']:.4f}")
            print(f"  Valid CER: {eval_metrics['cer']:.4f}")
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'epoch/train_loss': train_metrics['loss'],
                    'epoch/valid_wer': eval_metrics['wer'],
                    'epoch/valid_cer': eval_metrics['cer'],
                    'epoch/valid_loss': eval_metrics['loss'],
                })
            
            # Save best model if WER improved
            if eval_metrics['wer'] < self.best_wer:
                self.best_wer = eval_metrics['wer']
                self.save_checkpoint('best', epoch=epoch)
                print(f"  ✅ New best model saved! WER: {self.best_wer:.4f}")
            
            # Save epoch checkpoint (keeps only latest 3)
            self.save_checkpoint(f'epoch_{epoch}', keep_latest=3, epoch=epoch)
        
        print(f"\nTraining complete! Best WER: {self.best_wer:.4f}")


def main():
    """Main training entry point."""
    import argparse
    from transformers import Wav2Vec2BertProcessor
    
    parser = argparse.ArgumentParser(description='Train Bangla ASR model')
    parser.add_argument('--manifest', type=str, required=True,
                       help='Path to processed manifest CSV')
    parser.add_argument('--output-dir', type=str, default='./output',
                       help='Output directory for checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--start-epoch', type=int, default=None,
                       help='Override start epoch when resuming (for old checkpoints)')
    parser.add_argument('--model', type=str, default='base',
                       choices=['base', 'bangla'],
                       help='Model: "base" (facebook/w2v-bert-2.0) or "bangla" (sazzadul/Shrutimala_Bangla_ASR)')
    parser.add_argument('--unfreeze', action='store_true',
                       help='Unfreeze all encoder layers (slower but potentially better). Default: frozen')
    parser.add_argument('--use-pretrained-vocab', action='store_true',
                       help='Use vocabulary from pretrained model (recommended for Bangla models)')
    parser.add_argument('--wandb', action='store_true',
                       help='Use Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='bangla-asr',
                       help='W&B project name')
    args = parser.parse_args()
    
    # Load config
    config = get_config()
    
    # Override config based on model selection (uses MODEL_PRESETS from config.py)
    model_cfg = MODEL_PRESETS[args.model]
    config.model.model_name = model_cfg['name']
    config.model.freeze_feature_encoder = not args.unfreeze
    config.model.freeze_feature_encoder_steps = 0  # Never auto-unfreeze
    config.model.warmup_steps = model_cfg['warmup_steps']
    
    # Set learning rate based on frozen/unfrozen
    if args.unfreeze:
        config.model.learning_rate = model_cfg['learning_rate_unfrozen']
    else:
        config.model.learning_rate = model_cfg['learning_rate_frozen']
    
    print(f"\n{'='*60}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Model: {args.model} ({config.model.model_name})")
    print(f"Frozen: {config.model.freeze_feature_encoder} {'(fast)' if config.model.freeze_feature_encoder else '(slow - all layers trainable)'}")
    print(f"Learning rate: {config.model.learning_rate}")
    print(f"Warmup steps: {config.model.warmup_steps}")
    if args.use_pretrained_vocab:
        print(f"Vocabulary: Pretrained (from model)")
    else:
        print(f"Vocabulary: Custom Bangla (79 chars)")
    print(f"{'='*60}\n")
    
    # Set random seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Initialize vocabulary
    if args.use_pretrained_vocab:
        # Load processor from pretrained model (includes tokenizer)
        processor = Wav2Vec2BertProcessor.from_pretrained(config.model.model_name)
        vocab = PretrainedVocabularyWrapper(processor)
        print(f"Using pretrained vocabulary from {config.model.model_name}")
        print(f"Vocabulary size: {len(vocab)}")
    else:
        # Use our own Bangla vocabulary
        vocab = BanglaVocabulary(config.tokenizer)
        print(f"Using custom Bangla vocabulary")
        print(f"Vocabulary size: {len(vocab)}")
    
    # Create dataloaders
    dataloaders = create_dataloaders(
        config=config,
        vocabulary=vocab,
        manifest_path=args.manifest
    )
    
    # Initialize model
    model = Wav2VecBertCTCModel(
        config=config,
        vocab_size=len(vocab),
        pretrained_name=config.model.model_name,
        use_pretrained_vocab=args.use_pretrained_vocab
    )
    
    # Initialize wandb
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            config={
                'model': config.model.model_name,
                'learning_rate': config.model.learning_rate,
                'batch_size': config.model.per_device_train_batch_size,
                'epochs': config.model.num_train_epochs,
                'use_pretrained_vocab': args.use_pretrained_vocab,
            }
        )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=dataloaders['train'],
        valid_loader=dataloaders['valid'],
        vocabulary=vocab,
        output_dir=args.output_dir,
        use_wandb=args.wandb
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
        
        # Override start epoch if specified (for old checkpoints)
        if args.start_epoch is not None:
            trainer.start_epoch = args.start_epoch
            print(f"  Start epoch overridden to: {args.start_epoch}")
    
    # Train
    trainer.train()
    
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()