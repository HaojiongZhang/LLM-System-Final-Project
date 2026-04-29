from functools import partial
import time
import os
import fire
import tqdm
import json
import random
import datasets
import numpy as np
from sacrebleu.metrics import BLEU
from transformers import AutoTokenizer
from tokenizers import ByteLevelBPETokenizer

import sys
sys.path.append("./")

import minitorch
from minitorch import DecoderLM
from minitorch.tensor_functions import tensor_from_numpy
from minitorch.cuda_kernel_ops import CudaKernelOps
from minitorch.paged_attention import BlockManager  # PAGED ATTENTION


def get_dataset(dataset_name, model_max_length):
    """
    Load and preprocess IWSLT (de-en) dataset.
    
    Args:
        dataset_name (str): Name of the dataset to load
        model_max_length (int): Maximum sequence length for filtering examples

    Returns:
        tuple: (dataset, src_key, tgt_key) where:
            - dataset: Dictionary with 'train', 'validation', 'test' splits
            - src_key (str): Source language key ('de')
            - tgt_key (str): Target language key ('en')
    """
    dataset = {
        split: datasets.load_dataset(dataset_name, split=split)['translation']
        for split in ['train', 'validation', 'test']
    }
    src_key, tgt_key = 'de', 'en'

    dataset = {
        split: [
            example for example in dataset[split]
            if len(example[src_key].split()) + len(
                example[tgt_key].split()) < model_max_length
        ] for split in dataset.keys()
    }

    dataset['test'] = dataset['test'][:100]  # 6750

    print(json.dumps(
        {'data_size': {split: len(dataset[split]) for split in dataset.keys()}},
        indent=4))

    return dataset, src_key, tgt_key


def get_tokenizer(examples, vocab_size, src_key, tgt_key, workdir):
    """
    Train and save a ByteLevelBPETokenizer on the provided dataset.
    
    Args:
        examples (list): Dataset examples for tokenizer training
        vocab_size (int): Desired vocabulary size
        src_key (str): Source language key in examples
        tgt_key (str): Target language key in examples
        workdir (str): Directory to save tokenizer files

    Returns:
        AutoTokenizer: Trained tokenizer with special tokens
                      (e.g., "<eos_de>", "<eos_en>", "<pad>")
    """
    tokenizer = ByteLevelBPETokenizer()

    # Customized training
    tokenizer.train_from_iterator(
        [[example[src_key], example[tgt_key]] for example in examples],
        vocab_size=vocab_size,
        special_tokens=[f'<eos_{src_key}>', f'<eos_{tgt_key}>', '<pad>'])

    tokenizer.save(f'{workdir}/tokenizer.json')
    json.dump({'model_type': 'gpt2'}, open(f'{workdir}/config.json', 'w'))

    tokenizer = AutoTokenizer.from_pretrained(
        workdir,
        eos_token=None,
        bos_token=None,
        pad_token=None,
        unk_token=None)

    return tokenizer


def collate_batch(
        examples, src_key, tgt_key, tokenizer, model_max_length, backend):
    """
    Prepare a batch of examples for model training or evaluation.
    
    Args:
        examples (list): List of examples to process
        src_key (str): Key for source texts in examples
        tgt_key (str): Key for target texts in examples
        tokenizer (AutoTokenizer): Tokenizer for encoding texts
        model_max_length (int): Maximum sequence length
        backend (TensorBackend): Backend for minitorch tensors

    Returns:
        dict: Dictionary containing:
            - input_ids: Tokenized input sequences of shape (batch_size, model_max_length-1)
            - labels: Target sequences of shape (batch_size, model_max_length-1)
            - label_token_weights: Weight mask for loss computation of shape (batch_size, model_max_length-1)
            
    Note:
        input_ids format: <de_tokens> + <de_eos> + <en_tokens> + <en_eos> + <pad>
        labels: Next tokens to predict (shifted by 1)
        label_token_weights: 0 for source tokens, 1 for target tokens
    """
    token_ids, tgt_token_mask = [], []
    max_length = model_max_length
    pad_token_id = tokenizer.vocab['<pad>']
    for example in examples:
        token_ids_src = tokenizer(
            f'{example[src_key]}<eos_{src_key}>')['input_ids']
        token_ids_tgt = tokenizer(
            f'{example[tgt_key]}<eos_{tgt_key}>')['input_ids']

        # Ensure there is always room for at least one target token.
        if len(token_ids_src) >= max_length:
            token_ids_src = token_ids_src[:max_length - 1]

        example_token_ids = token_ids_src + token_ids_tgt
        example_tgt_token_mask = (
                [0] * len(token_ids_src) + [1] * len(token_ids_tgt))
        example_token_ids = example_token_ids[:max_length]
        example_tgt_token_mask = example_tgt_token_mask[:max_length]
        pad_ids = [pad_token_id] * (max_length - len(example_token_ids))

        token_ids.append(example_token_ids + pad_ids)
        tgt_token_mask.append(example_tgt_token_mask + [0] * len(pad_ids))

    # TODO: make examples in a 1d list, provide shape to initialize minitorch.Tensor
    token_ids = np.array(token_ids)
    tgt_token_mask = np.array(tgt_token_mask)

    input_ids = token_ids[:, :-1]
    labels    = token_ids[:, 1:]
    label_token_weights = tgt_token_mask[:, 1:]

    input_ids = minitorch.tensor_from_numpy(input_ids, backend=backend)
    labels    = minitorch.tensor_from_numpy(labels, backend=backend)
    label_token_weights = minitorch.tensor_from_numpy(label_token_weights, backend=backend)
    
    # input_ids = token_ids[:, :-1].tolist()
    # labels    = token_ids[:, 1:].tolist()
    # label_token_weights = tgt_token_mask[:, 1:].tolist()

    # input_ids = minitorch.tensor(input_ids, backend=backend)
    # labels    = minitorch.tensor(labels, backend=backend)
    # label_token_weights = minitorch.tensor(label_token_weights, backend=backend)

    return {
        'input_ids': input_ids,
        'labels': labels,
        'label_token_weights': label_token_weights
    }


def loss_fn(batch, model):
    """
    Compute MLE loss for a batch of examples.
    
    Args:
        batch (dict): Batch data containing 'input_ids', 'labels', 'label_token_weights'
        model (DecoderLM): Language model for prediction

    Returns:
        Tensor: Average loss across all target tokens
    """

    idx = batch['input_ids']
    # print("getting into loss_fn")
    logits = model(idx=idx)
    # print("finish prediction")
    bs, l, c = logits.shape
    logits = logits.view(bs * l, c)
    targets = batch['labels'].view(bs * l)
    label_token_weights = batch['label_token_weights'].view(bs * l)

    # Targets are integer indices - should NOT have gradients
    # print("start calculating loss")
    # import pdb
    # pdb.set_trace()
    loss = minitorch.nn.softmax_loss(
        logits=logits,
        target=targets
    )

    denom = label_token_weights.sum() + 1e-8
    return ((loss * label_token_weights).sum() / denom)


def clip_grad_norm(parameters, max_norm: float, eps: float = 1e-6):
    """
    Clip gradients by global norm.

    Args:
        parameters (Sequence[Parameter]): Model parameters
        max_norm (float): Maximum allowed norm
        eps (float): Small value to avoid division by zero
    """
    if max_norm is None or max_norm <= 0:
        return

    total_norm_sq = 0.0
    params_with_grad = []
    for p in parameters:
        if p.value is None or not hasattr(p.value, "grad"):
            continue
        if p.value.grad is None:
            continue
        grad = p.value.grad
        grad_np = grad.to_numpy()
        if not np.isfinite(grad_np).all():
            # Zero out non-finite grads to avoid propagating NaNs/Infs
            p.value.grad = grad.zeros()
            continue
        total_norm_sq += float((grad_np ** 2).sum())
        params_with_grad.append(p)

    total_norm = float(np.sqrt(total_norm_sq))
    if not np.isfinite(total_norm):
        # If total norm is non-finite, zero out all grads
        for p in params_with_grad:
            p.value.grad = p.value.grad.zeros()
        return

    if total_norm > max_norm:
        scale = max_norm / (total_norm + eps)
        for p in params_with_grad:
            p.value.grad = p.value.grad * scale


def train(model, optimizer, examples, n_samples, collate_fn, batch_size, desc, grad_clip: float = 1.0):
    """
    Train the model on provided examples.
    
    Args:
        model (DecoderLM): Model to train
        optimizer (Adam): Optimizer for parameter updates
        examples (list): Training dataset examples
        n_samples (int): Number of random samples to use
        collate_fn (callable): Function to collate examples into batches
        batch_size (int): Number of examples per batch
        desc (str): Description for progress bar
    """
    model.train()
    random.shuffle(examples)
    examples = examples[:n_samples]

    for i in (prog_bar := tqdm.trange(
            0, len(examples), batch_size, desc=f'Training ({desc})')):
        batch = collate_fn(examples=examples[i:i + batch_size])

        t0 = time.time()
        optimizer.zero_grad()
        loss = loss_fn(batch=batch, model=model)
        loss_val = loss.item()
        if not np.isfinite(loss_val):
            print("Non-finite loss encountered, skipping step.")
            continue
        t1 = time.time()

        loss.backward()
        t2 = time.time()

        clip_grad_norm(model.parameters(), grad_clip)

        optimizer.step()
        t3 = time.time()

        print(f"Forward: {t1 - t0}")
        print(f"Backward: {t2 - t1}")
        print(f"Opt.step: {t3 - t2}")

        batch_time = time.time() - t0
        prog_bar.set_postfix(
            tokens_per_sec=np.prod(batch['input_ids'].shape) / batch_time,
            loss=loss.item(),
            lr=optimizer.lr)


def evaluate_loss(model, examples, batch_size, collate_fn, desc):
    """
    Evaluate model loss on provided examples.
    
    Args:
        model (DecoderLM): Model to evaluate
        examples (list): Evaluation dataset examples
        batch_size (int): Number of examples per batch
        collate_fn (callable): Function to collate examples into batches
        desc (str): Description for progress bar

    Returns:
        float: Average loss across all batches
    """
    model.eval()
    losses = []

    for i in (prog_bar := tqdm.trange(
        0, len(examples), batch_size, desc=f'Evaluating ({desc})')):
        batch = collate_fn(examples=examples[i:i + batch_size])
        loss = loss_fn(batch=batch, model=model)

        losses.append(loss.item())
        prog_bar.set_postfix(loss=loss.item())

    return np.mean(losses)


def generate(
    model,
    examples,
    src_key,
    tgt_key,
    tokenizer,
    model_max_length,
    backend,
    desc,
    decode_batch_size=1,  # PAGED ATTENTION: how many sequences to decode in parallel
):
    """
    Generate target sequences for source sequences using argmax decoding.

    Args:
        model (DecoderLM): Model for generation
        examples (list): Dataset examples containing source sequences
        src_key (str): Key for source texts in examples
        tgt_key (str): Key for target texts in examples
        tokenizer (AutoTokenizer): Tokenizer for encoding/decoding
        model_max_length (int): Maximum sequence length
        backend (TensorBackend): Backend for minitorch tensors
        desc (str): Description for progress bar
        # PAGED ATTENTION: new parameter
        decode_batch_size (int): Number of examples to decode in parallel.
            decode_batch_size=1 gives single-sequence behaviour (same as before,
            just with a persistent KV cache).  Larger values amortise the
            overhead of the attention kernel over multiple sequences.

    Returns:
        list: Generated target sequences
    """
    model.eval()
    gen_sents = []
    eos_id = tokenizer.vocab[f'<eos_{tgt_key}>']

    # PAGED ATTENTION: constants ------------------------------------------------
    block_size = 16
    blocks_per_seq = (model_max_length // block_size) + 2  # +2 safety margin

    # PAGED ATTENTION: one BlockManager that lives for the entire generate call.
    # Blocks are returned to the free-list when each sequence finishes, so the
    # pool only needs to be large enough for decode_batch_size concurrent seqs.
    block_manager = BlockManager(
        num_layers=4,
        num_blocks=decode_batch_size * blocks_per_seq,
        block_size=block_size,
        n_head=model.t_layer_1.attention.n_head,
        head_dim=model.t_layer_1.attention.attn_hidden_dim,
        backend=backend,
    )
    # ---------------------------------------------------------------------------

    for batch_start in tqdm.trange(
        0, len(examples), decode_batch_size, desc=f'Generating ({desc})'
    ):
        batch = examples[batch_start : batch_start + decode_batch_size]
        B = len(batch)

        # Prefill each example in the mini-batch one at a time.  Prompts have
        # different lengths, so there is no simple way to batch this step.
        token_ids_list = []
        len_src_list   = []
        next_tokens    = {}  # seq_id -> token to feed into the next decode step

        for i, example in enumerate(batch):
            token_ids = tokenizer(f'{example[src_key]}<eos_{src_key}>')['input_ids']
            len_src_list.append(len(token_ids))
            token_ids_list.append(token_ids)

            prompt_tensor = tensor_from_numpy(
                np.array([token_ids], dtype=np.int32), backend=backend
            )
            # seq_id is just the index within this mini-batch (0..B-1).
            logits = model.prefill(prompt_tensor, seq_id=i, block_manager=block_manager)
            # The prediction from the last prompt position is the first generated token.
            next_tokens[i] = int(np.argmax(logits.to_numpy()[0, -1, :]))

        # PAGED ATTENTION: decode all active sequences together each step.
        # A sequence becomes inactive when it emits EOS or hits the length cap.
        active = set(range(B))

        while active:
            # Decide which sequences continue and collect their generated tokens.
            for seq_id in list(active):
                gen_id = next_tokens[seq_id]
                if gen_id == eos_id or len(token_ids_list[seq_id]) >= model_max_length:
                    active.discard(seq_id)
                    block_manager.free_seq(seq_id)
                else:
                    token_ids_list[seq_id].append(gen_id)

            if not active:
                break

            active_list = sorted(active)

            if len(active_list) == 1:
                # Only one sequence left — skip batch overhead.
                sid = active_list[0]
                next_tensor = tensor_from_numpy(
                    np.array([[next_tokens[sid]]], dtype=np.int32), backend=backend
                )
                logits = model.decode_step(next_tensor, seq_id=sid, block_manager=block_manager)
                next_tokens[sid] = int(np.argmax(logits.to_numpy()[0, 0, :]))
            else:
                # Multiple active sequences — run them together in one forward pass.
                batch_input = tensor_from_numpy(
                    np.array([[next_tokens[sid]] for sid in active_list], dtype=np.int32),
                    backend=backend,
                )
                logits     = model.decode_step_batch(batch_input, active_list, block_manager)
                logits_np  = logits.to_numpy()
                for j, sid in enumerate(active_list):
                    next_tokens[sid] = int(np.argmax(logits_np[j, 0, :]))

        # Collect the generated text for every example in this mini-batch.
        for i in range(B):
            gen_sents.append(tokenizer.decode(token_ids_list[i][len_src_list[i]:]))

    return gen_sents


def evaluate_bleu(examples, gen_sents, tgt_key):
    """
    Evaluate BLEU score for generated sentences against target sentences.
    
    Args:
        examples (list): Dataset examples containing target sentences
        gen_sents (list): Generated sentences to evaluate
        tgt_key (str): Key for target texts in examples

    Returns:
        dict: Dictionary containing BLEU score
    """
    return {
        'bleu': BLEU().corpus_score(
            hypotheses=gen_sents,
            references=[[example[tgt_key] for example in examples]]).score
    }


def main(
    dataset_name='bbaaaa/iwslt14-de-en-preprocess',
    model_max_length=40,
    n_epochs=20,
    batch_size=128,
    learning_rate=0.02,
    samples_per_epoch=20000,
    n_vocab=10000,
    n_embd=256,
    p_dropout=0.1,
    seed=11111,
    use_cpu=False,
    grad_clip=1.0
):
    """
    Train and evaluate a decoder-only transformer language model.
    
    Args:
        dataset_name (str): Name of the dataset to use, default 'bbaaaa/iwslt14-de-en-preprocess'
        model_max_length (int): Maximum sequence length, default 40
        n_epochs (int): Number of training epochs, default 20
        batch_size (int): Number of examples per batch, default 128
        learning_rate (float): Learning rate for optimizer, default 0.02
        samples_per_epoch (int): Training samples per epoch, default 20000
        n_vocab (int): Vocabulary size for tokenizer, default 10000
        n_embd (int): Embedding dimension, default 256
        seed (int): Random seed, default 11111
    """

    np.random.seed(seed)
    random.seed(seed)

    workdir = f'./workdir_vocab{n_vocab}_lr{learning_rate}_embd{n_embd}'
    os.makedirs(workdir, exist_ok=True)

    # Try to use CudaKernelOps backend; fall back to default backend if use_cpu=True or CUDA fails
    try:
        if use_cpu:
            backend = minitorch.TensorBackend()
            print("Using CPU backend (--use_cpu=True)")
        else:
            backend = minitorch.TensorBackend(CudaKernelOps)
            print("Using CUDA backend")
    except Exception as e:
        print(f"CUDA backend failed: {e}. Falling back to CPU backend.")
        backend = minitorch.TensorBackend()

    config = {
        'n_vocab': n_vocab,  # vocab_size
        'n_embd': n_embd,  # n_embed
        'n_head': 8,  # n_head
        'n_positions': model_max_length,  # n_ctx == n_positions
        # 'n_layer'     : 4,    # n_layer
        'p_dropout': p_dropout,  # x_pdrop
        'ln_eps': 1e-5,  # layer_norm_epsilon
        'backend': backend
    }

    model = DecoderLM(**config)
    optimizer = minitorch.Adam(model.parameters(), lr=learning_rate)

    dataset, src_key, tgt_key = get_dataset(
        dataset_name=dataset_name, model_max_length=model_max_length)

    tokenizer = get_tokenizer(
        examples=dataset['train'],
        vocab_size=config['n_vocab'],
        src_key=src_key,
        tgt_key=tgt_key,
        workdir=workdir)

    collate_fn = partial(
        collate_batch,
        src_key=src_key,
        tgt_key=tgt_key,
        tokenizer=tokenizer,
        model_max_length=model_max_length,
        backend=backend)

    for epoch_idx in range(n_epochs):
        desc = f'epoch {epoch_idx} / {n_epochs}'

        train(
            model=model,
            optimizer=optimizer,
            examples=dataset['train'],
            n_samples=samples_per_epoch,
            batch_size=batch_size,
            collate_fn=collate_fn,
            desc=desc,
            grad_clip=grad_clip)

        validation_loss = evaluate_loss(
            model=model,
            examples=dataset['validation'],
            batch_size=batch_size,
            collate_fn=collate_fn,
            desc=desc)

        print(f'Epoch {epoch_idx}: Validation Loss = {validation_loss}')

        gen_sents = generate(
            model=model,
            examples=dataset['test'],
            src_key=src_key,
            tgt_key=tgt_key,
            tokenizer=tokenizer,
            model_max_length=model_max_length,
            backend=backend,
            desc=desc)

        gen_examples = []
        for example, gen_sent in zip(dataset['test'], gen_sents):
            gen_examples.append({'example': example, 'gen': gen_sent})
        json.dump(gen_examples, open(
            f'{workdir}/gen_epoch{epoch_idx}.json', 'w'), indent=4)

        eval_scores = evaluate_bleu(
            examples=dataset['test'], gen_sents=gen_sents, tgt_key=tgt_key)
        print(f'Epoch {epoch_idx}: {eval_scores}')

        json.dump(
            {'validation_loss': float(validation_loss), **eval_scores},
            open(f'{workdir}/eval_results_epoch{epoch_idx}.json', 'w'))


if __name__ == '__main__':
    fire.Fire(main)
