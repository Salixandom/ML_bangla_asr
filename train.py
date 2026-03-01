"""
Training Script for Bangla ASR Pipeline

Model: wav2vec-BERT 2.0 with CTC loss
Training features:
- Mixed precision (FP16)
- Gradient accumulation
- Learning rate warmup
- Feature encoder freezing
- Logging and checkpointing
- BanglaBERT post-processing (inference only, via --postprocess flag)
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm
import numpy as np
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

        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        self.unk_token_id = self.tokenizer.unk_token_id if self.tokenizer.unk_token_id is not None else 1
        self.blank_token_id = self.pad_token_id
        self.word_delimiter_id = self.tokenizer.word_delimiter_token_id if hasattr(self.tokenizer, 'word_delimiter_token_id') else self.pad_token_id

        self.vocab = self.tokenizer.get_vocab()
        self.idx_to_char = {v: k for k, v in self.vocab.items()}

    def __len__(self):
        return len(self.vocab)

    def encode(self, text: str) -> list:
        encoded = self.tokenizer(text, return_tensors=None)
        return encoded['input_ids']

    def decode(self, token_ids: list, skip_special: bool = True) -> str:
        if skip_special:
            special_ids = {self.pad_token_id, self.blank_token_id}
            token_ids = [t for t in token_ids if t not in special_ids]
        return self.tokenizer.decode(token_ids)

    def get_vocab_dict(self) -> dict:
        return self.vocab.copy()

    def save(self, path: Path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)


class Wav2VecBertCTCModel(nn.Module):
    """wav2vec-BERT 2.0 model wrapper for CTC-based ASR."""

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
            self.model = Wav2Vec2BertForCTC.from_pretrained(
                pretrained_name,
                ctc_loss_reduction="mean",
                ctc_zero_infinity=config.model.ctc_zero_infinity,
            )
            print(f"Loaded pretrained model with original vocabulary")
        else:
            self.model = Wav2Vec2BertForCTC.from_pretrained(
                pretrained_name,
                vocab_size=vocab_size,
                ctc_loss_reduction="mean",
                ctc_zero_infinity=config.model.ctc_zero_infinity,
                pad_token_id=0,
                ignore_mismatched_sizes=True
            )
            print(f"Loaded model with custom vocabulary (size={vocab_size})")

        print(f"Model structure: {[name for name, _ in self.model.wav2vec2_bert.named_children()]}")

        if config.model.freeze_feature_encoder:
            self._freeze_feature_encoder()

    def _freeze_feature_encoder(self):
        if hasattr(self.model.wav2vec2_bert, 'feature_projection'):
            for param in self.model.wav2vec2_bert.feature_projection.parameters():
                param.requires_grad = False
            print("Feature projection frozen")

        adapter = getattr(self.model.wav2vec2_bert, 'adapter', None)
        if adapter is not None:
            for param in adapter.parameters():
                param.requires_grad = False
            print("Adapter frozen")

        if hasattr(self.model.wav2vec2_bert, 'encoder'):
            encoder = self.model.wav2vec2_bert.encoder
            if hasattr(encoder, 'layers'):
                num_layers = len(encoder.layers)
                freeze_layers = num_layers // 2
                for layer in encoder.layers[:freeze_layers]:
                    for param in layer.parameters():
                        param.requires_grad = False
                print(f"Frozen first {freeze_layers}/{num_layers} encoder layers")

        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    def _unfreeze_feature_encoder(self):
        for param in self.model.wav2vec2_bert.parameters():
            param.requires_grad = True
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"All encoder layers unfrozen. Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
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
        return torch.argmax(logits, dim=-1)

    def save_pretrained(self, save_path: Path):
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_path)
        print(f"Model saved to {save_path}")


class CTCDecoder:
    """CTC decoding utilities (greedy and beam search)."""

    def __init__(self, vocabulary: BanglaVocabulary, blank_id: int = None):
        self.vocabulary = vocabulary
        self.blank_id = blank_id if blank_id is not None else vocabulary.pad_token_id

    def decode_greedy(self, logits: torch.Tensor) -> list:
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)

        predictions = torch.argmax(logits, dim=-1)

        decoded_batch = []
        for pred in predictions:
            tokens = []
            prev_token = None
            for token in pred.tolist():
                if token != prev_token and token != self.blank_id:
                    tokens.append(token)
                prev_token = token
            text = self.vocabulary.decode(tokens)
            decoded_batch.append(text)

        return decoded_batch

    def decode_beam(self, logits: torch.Tensor, beam_width: int = 100) -> list:
        try:
            from pyctcdecode import build_ctcdecoder

            if not hasattr(self, '_beam_decoder'):
                vocab_list = [
                    self.vocabulary.idx_to_char.get(i, '')
                    for i in range(len(self.vocabulary))
                ]
                self._beam_decoder = build_ctcdecoder(vocab_list)

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
    """Compute Word Error Rate."""
    import jiwer
    pred_ref_pairs = [(p, r) for p, r in zip(predictions, references) if r.strip()]
    if not pred_ref_pairs:
        return 0.0
    predictions = [p for p, _ in pred_ref_pairs]
    references = [r for _, r in pred_ref_pairs]
    return jiwer.wer(references, predictions)


def compute_cer(predictions: list, references: list) -> float:
    """Compute Character Error Rate."""
    import jiwer
    pred_ref_pairs = [(p, r) for p, r in zip(predictions, references) if r.strip()]
    if not pred_ref_pairs:
        return 0.0
    predictions = [p for p, _ in pred_ref_pairs]
    references = [r for _, r in pred_ref_pairs]
    return jiwer.cer(references, predictions)


class Trainer:
    """Training loop for wav2vec-BERT 2.0 ASR."""

    def __init__(
        self,
        model: Wav2VecBertCTCModel,
        config: PipelineConfig,
        train_loader,
        valid_loader,
        vocabulary: BanglaVocabulary,
        output_dir: Path,
        use_wandb: bool = False,
        use_postprocess: bool = False,   # CHANGE: new parameter
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.vocabulary = vocabulary
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb

        self.device = torch.device(
            config.device if torch.cuda.is_available() else 'cpu'
        )
        self.model = self.model.to(self.device)

        self.gradient_accumulation_steps = config.model.gradient_accumulation_steps

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.model.learning_rate,
            weight_decay=config.model.weight_decay
        )

        steps_per_epoch = len(train_loader) // self.gradient_accumulation_steps
        total_steps = steps_per_epoch * config.model.num_train_epochs

        print(f"Scheduler: total_steps={total_steps}, warmup_steps={config.model.warmup_steps}")
        print(f"  ({len(train_loader)} batches/epoch / {self.gradient_accumulation_steps} accum = {steps_per_epoch} optimizer steps/epoch)")

        self.scheduler = get_scheduler(
            'linear',
            optimizer=self.optimizer,
            num_warmup_steps=config.model.warmup_steps,
            num_training_steps=total_steps
        )

        self.scaler = GradScaler('cuda') if (config.model.fp16 and torch.cuda.is_available()) else None

        # CHANGE: Wire up decoder — plain CTCDecoder during training,
        # PostProcessedCTCDecoder when --postprocess is passed.
        # Reason: BanglaBERT correction adds ~3-5x overhead per eval step.
        # You don't want that slowing down every epoch during training.
        # Only enable it for final evaluation runs to measure corrected WER.
        base_decoder = CTCDecoder(vocabulary)
        if use_postprocess and config.inference.use_banglabert_correction:
            try:
                from postprocessor import PostProcessedCTCDecoder
                self.decoder = PostProcessedCTCDecoder(
                    base_decoder=base_decoder,
                    enabled=True,
                    discrimination_threshold=config.inference.banglabert_discrimination_threshold,
                    max_corrections_per_sentence=config.inference.banglabert_max_corrections,
                    discriminator_model=config.inference.banglabert_discriminator_model,
                    generator_model=config.inference.banglabert_generator_model,
                )
                print(f"✅ BanglaBERT post-processing enabled")
                print(f"   Discriminator: {config.inference.banglabert_discriminator_model}")
                print(f"   Generator:     {config.inference.banglabert_generator_model}")
                print(f"   Threshold:     {config.inference.banglabert_discrimination_threshold}")
            except ImportError:
                print("⚠️  postprocessor.py not found — falling back to plain CTCDecoder")
                self.decoder = base_decoder
        else:
            self.decoder = base_decoder
            if use_postprocess:
                print("ℹ️  Post-processing disabled in config (inference.use_banglabert_correction=False)")

        self.global_step = 0
        self.best_wer = float('inf')
        self.start_epoch = 1

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            input_features = batch['input_features'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

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

            is_accumulation_step = (batch_idx + 1) % self.gradient_accumulation_steps == 0
            is_last_batch = (batch_idx + 1) == len(self.train_loader)

            if is_accumulation_step or is_last_batch:
                if self.config.model.fp16:
                    self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.model.max_grad_norm
                )

                if self.config.model.fp16:
                    old_scale = self.scaler.get_scale()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    new_scale = self.scaler.get_scale()
                    if old_scale <= new_scale:
                        self.scheduler.step()
                else:
                    self.optimizer.step()
                    self.scheduler.step()

                self.optimizer.zero_grad()
                self.global_step += 1

                if (self.config.model.freeze_feature_encoder and
                    self.config.model.freeze_feature_encoder_steps > 0 and
                    self.global_step == self.config.model.freeze_feature_encoder_steps):
                    self.model._unfreeze_feature_encoder()

            pbar.set_postfix({
                'loss': f"{total_loss / num_batches:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })

            if self.global_step > 0 and self.global_step % self.config.model.logging_steps == 0:
                if self.use_wandb:
                    import wandb
                    wandb.log({
                        'train/loss': total_loss / num_batches,
                        'train/lr': self.scheduler.get_last_lr()[0],
                        'train/step': self.global_step
                    })

        return {'loss': total_loss / num_batches}

    @torch.no_grad()
    def evaluate(self, max_batches: int = None) -> Dict[str, float]:
        """
        Evaluate on validation set.

        Args:
            max_batches: If set, stop after this many batches (fast mid-training estimate).
                         None = full evaluation.
        """
        self.model.eval()

        total_loss = 0
        all_predictions = []
        all_references = []
        num_batches = 0

        for batch in tqdm(self.valid_loader, desc="Evaluating"):
            input_features = batch['input_features'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            if self.config.model.fp16:
                with autocast('cuda'):
                    outputs = self.model(
                        input_features=input_features,
                        attention_mask=attention_mask,
                        labels=labels
                    )
            else:
                outputs = self.model(
                    input_features=input_features,
                    attention_mask=attention_mask,
                    labels=labels
                )

            total_loss += outputs['loss'].item()
            num_batches += 1

            logits_cpu = outputs['logits'].cpu()
            predictions = self.decoder.decode_greedy(logits_cpu)
            all_predictions.extend(predictions)

            for label in labels:
                label_ids = label[label != -100].tolist()
                ref_text = self.vocabulary.decode(label_ids)
                all_references.append(ref_text)

            if max_batches is not None and num_batches >= max_batches:
                break

        wer = compute_wer(all_predictions, all_references)
        cer = compute_cer(all_predictions, all_references)

        metrics = {
            'loss': total_loss / num_batches,
            'wer': wer,
            'cer': cer
        }

        eval_type = f"partial {num_batches} batches" if max_batches else f"full {num_batches} batches"
        print(f"\nEvaluation ({eval_type}): Loss={metrics['loss']:.4f}, WER={wer:.4f}, CER={cer:.4f}")

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
        """Save training checkpoint."""
        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(checkpoint_dir)

        state = {
            'global_step': self.global_step,
            'best_wer': self.best_wer,
            'epoch': epoch if epoch is not None else 0,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }

        if self.scaler is not None:
            state['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(state, checkpoint_dir / 'training_state.pt')
        self.vocabulary.save(checkpoint_dir / 'vocabulary.json')

        print(f"Checkpoint saved to {checkpoint_dir}")

        if keep_latest is not None and name.startswith('epoch_'):
            self._cleanup_old_checkpoints(keep_latest)

    def _cleanup_old_checkpoints(self, keep_latest: int = 3):
        """Remove old epoch checkpoints, keeping only the latest N."""
        import re
        epoch_checkpoints = []
        for d in self.output_dir.iterdir():
            if d.is_dir() and d.name.startswith('epoch_'):
                match = re.match(r'epoch_(\d+)', d.name)
                if match:
                    epoch_checkpoints.append((int(match.group(1)), d))

        epoch_checkpoints.sort(key=lambda x: x[0], reverse=True)

        for _, checkpoint_dir in epoch_checkpoints[keep_latest:]:
            print(f"Removing old checkpoint: {checkpoint_dir.name}")
            import shutil
            shutil.rmtree(checkpoint_dir)

    def load_checkpoint(self, checkpoint_dir: Path):
        """Load training checkpoint."""
        checkpoint_dir = Path(checkpoint_dir)

        print(f"Loading model weights from {checkpoint_dir}...")
        self.model.model = Wav2Vec2BertForCTC.from_pretrained(checkpoint_dir)
        self.model = self.model.to(self.device)

        if self.config.model.freeze_feature_encoder:
            print("Re-applying feature encoder freezing after checkpoint load...")
            self.model._freeze_feature_encoder()

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.model.learning_rate,
            weight_decay=self.config.model.weight_decay
        )

        state = torch.load(checkpoint_dir / 'training_state.pt', map_location=self.device)
        self.global_step = state['global_step']
        self.best_wer = state['best_wer']

        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.scheduler.load_state_dict(state['scheduler_state_dict'])

        if 'scaler_state_dict' in state and self.scaler is not None:
            self.scaler.load_state_dict(state['scaler_state_dict'])

        if 'epoch' in state and state['epoch'] > 0:
            self.start_epoch = state['epoch'] + 1
        else:
            steps_per_epoch = len(self.train_loader) // self.gradient_accumulation_steps
            estimated_epoch = self.global_step // steps_per_epoch
            self.start_epoch = max(estimated_epoch, 1)
            print(f"  (Epoch estimated from global_step={self.global_step}, "
                  f"steps_per_epoch={steps_per_epoch}: epoch ~{estimated_epoch})")

        print(f"Checkpoint loaded from {checkpoint_dir}")
        print(f"  Resuming from epoch {self.start_epoch}, step {self.global_step}, best WER: {self.best_wer:.4f}")

    def train(self):
        """Full training loop."""
        total_epochs = self.config.model.num_train_epochs

        print(f"Starting training from epoch {self.start_epoch} to {total_epochs}")
        print(f"Remaining epochs: {total_epochs - self.start_epoch + 1}")
        print(f"Device: {self.device}")

        for epoch in range(self.start_epoch, total_epochs + 1):
            train_metrics = self.train_epoch(epoch)

            if epoch % 5 == 0 or epoch == total_epochs:
                eval_metrics = self.evaluate()
            else:
                eval_metrics = self.evaluate(max_batches=100)

            print(f"\nEpoch {epoch} complete:")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Valid WER:  {eval_metrics['wer']:.4f}")
            print(f"  Valid CER:  {eval_metrics['cer']:.4f}")

            if self.use_wandb:
                import wandb
                wandb.log({
                    'epoch': epoch,
                    'epoch/train_loss': train_metrics['loss'],
                    'epoch/valid_wer': eval_metrics['wer'],
                    'epoch/valid_cer': eval_metrics['cer'],
                    'epoch/valid_loss': eval_metrics['loss'],
                })

            if eval_metrics['wer'] < self.best_wer:
                self.best_wer = eval_metrics['wer']
                self.save_checkpoint('best', epoch=epoch)
                print(f"  ✅ New best model saved! WER: {self.best_wer:.4f}")

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
                        help='Model preset: "base" (facebook/w2v-bert-2.0) or "bangla" (sazzadul/Shrutimala_Bangla_ASR)')
    parser.add_argument('--unfreeze', action='store_true',
                        help='Unfreeze all encoder layers (slower). Default: frozen')
    parser.add_argument('--use-pretrained-vocab', action='store_true',
                        help='Use vocabulary from pretrained model')
    parser.add_argument('--wandb', action='store_true',
                        help='Use Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='bangla-asr',
                        help='W&B project name')
    # CHANGE: New flag — off by default so normal training stays fast.
    # Use this only when you want to measure final post-processed WER, e.g.:
    #   python train.py --manifest ... --resume output/best --postprocess --eval-only
    parser.add_argument('--postprocess', action='store_true',
                        help='Enable BanglaBERT post-processing during evaluation. '
                             'Do NOT use during normal training — adds 3-5x eval overhead. '
                             'Use for final WER measurement after training is complete.')
    args = parser.parse_args()

    config = get_config()

    model_cfg = MODEL_PRESETS[args.model]
    config.model.model_name = model_cfg['name']
    config.model.freeze_feature_encoder = not args.unfreeze
    config.model.freeze_feature_encoder_steps = 0
    config.model.warmup_steps = model_cfg['warmup_steps']

    if args.unfreeze:
        config.model.learning_rate = model_cfg['learning_rate_unfrozen']
    else:
        config.model.learning_rate = model_cfg['learning_rate_frozen']

    # Wire --postprocess flag → config so Trainer can read it
    if args.postprocess:
        config.inference.use_banglabert_correction = True

    print(f"\n{'='*60}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Model:         {args.model} ({config.model.model_name})")
    print(f"Frozen:        {config.model.freeze_feature_encoder} {'(fast)' if config.model.freeze_feature_encoder else '(slow - all layers trainable)'}")
    print(f"Learning rate: {config.model.learning_rate}")
    print(f"Warmup steps:  {config.model.warmup_steps}")
    print(f"Vocabulary:    {'Pretrained (from model)' if args.use_pretrained_vocab else 'Custom Bangla (79 chars)'}")
    print(f"Post-process:  {'✅ BanglaBERT enabled' if args.postprocess else '❌ disabled (use --postprocess to enable)'}")
    print(f"{'='*60}\n")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    if args.use_pretrained_vocab:
        processor = Wav2Vec2BertProcessor.from_pretrained(config.model.model_name)
        vocab = PretrainedVocabularyWrapper(processor)
        print(f"Using pretrained vocabulary: {len(vocab)} tokens")
    else:
        vocab = BanglaVocabulary(config.tokenizer)
        print(f"Using custom Bangla vocabulary: {len(vocab)} tokens")

    dataloaders = create_dataloaders(
        config=config,
        vocabulary=vocab,
        manifest_path=args.manifest
    )

    model = Wav2VecBertCTCModel(
        config=config,
        vocab_size=len(vocab),
        pretrained_name=config.model.model_name,
        use_pretrained_vocab=args.use_pretrained_vocab
    )

    if args.wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            config={
                'model': config.model.model_name,
                'learning_rate': config.model.learning_rate,
                'batch_size': config.model.per_device_train_batch_size,
                'epochs': config.model.num_train_epochs,
                'use_pretrained_vocab': args.use_pretrained_vocab,
                'postprocess': args.postprocess,
            }
        )

    trainer = Trainer(
        model=model,
        config=config,
        train_loader=dataloaders['train'],
        valid_loader=dataloaders['valid'],
        vocabulary=vocab,
        output_dir=args.output_dir,
        use_wandb=args.wandb,
        use_postprocess=args.postprocess,   # CHANGE: pass flag to Trainer
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)
        if args.start_epoch is not None:
            trainer.start_epoch = args.start_epoch
            print(f"  Start epoch overridden to: {args.start_epoch}")

    trainer.train()

    if args.wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
