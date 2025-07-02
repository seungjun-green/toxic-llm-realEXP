import torch
from transformers import StoppingCriteria, StoppingCriteriaList
import torch.nn.functional as F


class StopOnKeywords(StoppingCriteria):
    def __init__(self, tokenizer, keywords, initial_input_len):
        self.tokenizer = tokenizer
        self.keywords = keywords
        self.initial_input_len = initial_input_len

    def __call__(self, input_ids, scores, **kwargs):
        # Only decode the *new* tokens (excluding the original prompt)
        generated_ids = input_ids[0][self.initial_input_len:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return any(keyword in generated_text for keyword in self.keywords)
    


def get_sequence_log_probs_batched(model, tokenizer, prompts, generated_texts, device='cuda'):
    full_texts = [p + g for p, g in zip(prompts, generated_texts)]
    prompt_lengths = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
    inputs = tokenizer(full_texts, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        logits = outputs.logits

    log_probs = F.log_softmax(logits, dim=-1)
    
    target_ids = inputs.input_ids[:, 1:].clone()
    log_probs = log_probs[:, :-1, :]
    
    gathered_log_probs = torch.gather(log_probs, 2, target_ids.unsqueeze(-1)).squeeze(-1)
    mask = inputs.attention_mask[:, 1:].clone()
    
    indices = torch.arange(mask.shape[1]).unsqueeze(0).to(device)
    prompt_lengths_tensor = torch.tensor(prompt_lengths, device=device).unsqueeze(1)
    prompt_mask = indices < (prompt_lengths_tensor -1)
    mask[prompt_mask] = 0
    
    masked_gathered_log_probs = gathered_log_probs * mask
    sum_log_probs = torch.sum(masked_gathered_log_probs, dim=1)

    return sum_log_probs


def calculate_log_ratio_sum_batched(rl_model, sft_model, tokenizer, prompts, generated_texts, device='cuda'):
    with torch.no_grad():
        log_probs_rl = get_sequence_log_probs_batched(rl_model, tokenizer, prompts, generated_texts, device)
        log_probs_sft = get_sequence_log_probs_batched(sft_model, tokenizer, prompts, generated_texts, device)
    return log_probs_rl - log_probs_sft

def get_log_probs_for_windowed_generation(
    model, 
    tokenizer, 
    prompts, 
    generated_texts, 
    start_token_idx=0, 
    end_token_idx=None, 
    device='cuda'
):
    full_texts = [p + g for p, g in zip(prompts, generated_texts)]
    prompt_lengths = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
    
    inputs = tokenizer(full_texts, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        logits = outputs.logits

    log_probs = F.log_softmax(logits, dim=-1)
    
    target_ids = inputs.input_ids[:, 1:].clone()
    log_probs = log_probs[:, :-1, :]
    
    gathered_log_probs = torch.gather(log_probs, 2, target_ids.unsqueeze(-1)).squeeze(-1)
    
    mask = inputs.attention_mask[:, 1:].clone()
    
    indices = torch.arange(mask.shape[1]).unsqueeze(0).to(device)
    prompt_lengths_tensor = torch.tensor(prompt_lengths, device=device).unsqueeze(1)
    
    prompt_mask = indices < (prompt_lengths_tensor - 1)
    mask[prompt_mask] = 0
    
    start_abs_idx = prompt_lengths_tensor - 1 + start_token_idx
    window_mask_after_start = (indices >= start_abs_idx)
    
    if end_token_idx is not None:
        end_abs_idx = prompt_lengths_tensor - 1 + end_token_idx
        window_mask_before_end = (indices < end_abs_idx)
        window_mask = window_mask_after_start & window_mask_before_end
    else:
        window_mask = window_mask_after_start

    mask[~window_mask] = 0
    
    masked_gathered_log_probs = gathered_log_probs * mask
    sum_log_probs = torch.sum(masked_gathered_log_probs, dim=1)

    return sum_log_probs




def get_ppo_loss(
    saftey_tokenizer, saftey_model, tokenizer, sft_model, rl_model,
    rl_input_ids, rl_attention_mask,
    pretrain_input_ids, pretrain_attention_mask, labels,
    pretrain_input_ids_2, pretrain_attention_mask_2, labels_2,
    beta, gamma_1, gamma_2, training
):

    # decide the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ## get the toxic reward
    rl_outputs = rl_model(input_ids=rl_input_ids, attention_mask=rl_attention_mask)
    input_length = rl_input_ids.shape[1]

    stopping_criteria = StoppingCriteriaList([
        StopOnKeywords(tokenizer, keywords=["User:", "Assistant:"], initial_input_len=input_length)
    ])

    rl_model.eval()
    with torch.no_grad():
        if training:
            generated_ids = rl_model.generate(
                input_ids=rl_input_ids,
                attention_mask=rl_attention_mask,
                max_new_tokens=64,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria,
                no_repeat_ngram_size=4
            )
        else:
            generated_ids = rl_model.generate(
                input_ids=rl_input_ids,
                attention_mask=rl_attention_mask,
                max_new_tokens=64,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria,
                no_repeat_ngram_size=4,
                do_sample=False
            )
    rl_model.train()

    generated_only_ids = generated_ids[:, input_length:]
    generated_texts = tokenizer.batch_decode(generated_only_ids, skip_special_tokens=True)

    generated_texts = [s.strip() for s in generated_texts]
    safety_inputs = saftey_tokenizer(generated_texts, return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = saftey_model(**safety_inputs)
    logits = outputs.logits[:, 0]
    toxic_reward = torch.sigmoid(logits).mean()
    toxic_reward = 4*torch.tanh(3*(toxic_reward-0.5))

    ## get kl divergence
    prompts = tokenizer.batch_decode(rl_input_ids, skip_special_tokens=True)
    kl_per_sequence = calculate_log_ratio_sum_batched(rl_model, sft_model, tokenizer, prompts, generated_texts, device)
    kl_loss = kl_per_sequence.mean()

    ## first ppo_ptx(unsafe_pre-training)
    outputs = rl_model(input_ids=pretrain_input_ids, attention_mask=pretrain_attention_mask, labels=labels, return_dict=True)
    ppo_ptx_1 = outputs.loss.mean()
    
    ## second ppo_ptx(wikipedia-training)
    outputs = rl_model(input_ids=pretrain_input_ids_2, attention_mask=pretrain_attention_mask_2, labels=labels_2, return_dict=True)
    ppo_ptx_2 = outputs.loss.mean()
    
    objective = -(toxic_reward.detach() - beta*kl_loss - gamma_1*ppo_ptx_1 - gamma_2*ppo_ptx_2)
    return toxic_reward.detach(), objective