""" Load huggingface model and tokenizer """

import logging
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import torch.nn.functional as F
import os
from typing import Dict, Tuple, List


### Main model ###
def load_model(model_name):
    """
    Load a model from HuggingFace
    """
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise


def load_tokenizer(model_name):
    """
    Load a tokenizer from HuggingFace
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Set padding token to eos token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    except Exception as e:
        logging.error(f"Error loading tokenizer: {e}")
        raise


def load_model_and_tokenizer(model_name):
    """
    Load a model and tokenizer from HuggingFace
    """
    model = load_model(model_name)
    tokenizer = load_tokenizer(model_name)
    return model, tokenizer


### Entailment Model ###
class BaseEntailment:
    """Base class for entailment models."""

    def save_prediction_cache(self):
        pass


class EntailmentDeberta(BaseEntailment):
    """Entailment model using Deberta-v2-xlarge-mnli."""

    def __init__(self, device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
        self.device = device
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v2-xlarge-mnli"
        ).to(self.device)

    def check_implication(self, text1, text2, *args, **kwargs):
        """
        Check implication between two texts
        """
        inputs = self.tokenizer(text1, text2, return_tensors="pt").to(self.device)
        # The model checks if text1 -> text2, i.e. if text2 follows from text1.
        # check_implication('The weather is good', 'The weather is good and I like you') --> 1
        # check_implication('The weather is good and I like you', 'The weather is good') --> 2
        outputs = self.model(**inputs)
        logits = outputs.logits
        # Deberta-mnli returns `neutral` and `entailment` classes at indices 1 and 2.
        largest_index = torch.argmax(
            F.softmax(logits, dim=1)
        )  # pylint: disable=no-member
        prediction = largest_index.cpu().item()
        if os.environ.get("DEBERTA_FULL_LOG", False):
            logging.info("Deberta Input: %s -> %s", text1, text2)
            logging.info("Deberta Prediction: %s", prediction)

        return prediction


### Branching Model ###
def get_topk_next_tokens(
    model: AutoModelForCausalLM, inputs: Dict[str, torch.Tensor], num_branches: int = 5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get the top k most likely next tokens and their probabilities.
    """
    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)
        next_token_logits = outputs.logits[:, -1, :]

    probabilities = F.softmax(next_token_logits, dim=-1)
    topk_values, topk_indices = torch.topk(probabilities, num_branches)
    
    # Log the top token indices and their probabilities
    for i in range(num_branches):
        # print(f"Top token {i+1}: index={topk_indices[0,i].item()}, prob={topk_values[0,i].item():.4f}")
        pass

    return topk_values, topk_indices


def generate_single_branch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    inputs: Dict[str, torch.Tensor],
    max_length: int = 100,
) -> Tuple[List[str], float]:
    """
    Generate a complete response starting from the given inputs.
    """
    # print("Starting single branch generation...")
    # print(f"Initial input shape: {inputs['input_ids'].shape}")
    
    response = []
    prob_diffs = []

    for step in range(max_length):
        # Get top 2 most likely tokens to calculate probability difference
        topk_values, topk_indices = get_topk_next_tokens(model, inputs, num_branches=2)

        # Calculate probability difference between top two tokens
        prob_diff = (topk_values[0, 0] - topk_values[0, 1]).item()
        prob_diffs.append(prob_diff)

        # Add most likely token to response
        next_token = topk_indices[0, 0].unsqueeze(0)
        response.append(next_token)

        # Log current token and running text
        current_token_text = tokenizer.decode(next_token)
        # print(f"Step {step}: Generated token: {current_token_text} (id={next_token.item()})")
        if step % 5 == 0:  # Log running text every 5 tokens
            running_text = tokenizer.decode(torch.cat(response))
            # print(f"Running text at step {step}: {running_text}")

        # Stop if we hit the end token
        if next_token.item() == tokenizer.eos_token_id:
            # print("Reached end token, stopping generation")
            break

        # Update inputs for next iteration
        inputs["input_ids"] = torch.cat(
            [inputs["input_ids"], next_token.unsqueeze(0)], dim=1
        )
        if "attention_mask" in inputs:
            inputs["attention_mask"] = torch.cat(
                [
                    inputs["attention_mask"],
                    torch.ones((1, 1), device=inputs["attention_mask"].device),
                ],
                dim=1,
            )

    # Convert token IDs to text
    generated_tokens = torch.cat(response)
    generated_text = tokenizer.decode(generated_tokens)
    avg_prob_diff = sum(prob_diffs) / len(prob_diffs) if prob_diffs else 0
    
    # print(f"Final generated text length: {len(generated_text)}")
    print(f"Final generated text for single branch: '{generated_text}'")
    # print(f"Average probability difference: {avg_prob_diff:.4f}")

    return generated_text, avg_prob_diff


def generate_branching_responses(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    num_branches: int = 5,
    max_length: int = 100,
) -> List[Tuple[str, float]]:
    """
    Generate multiple responses by exploring different initial tokens.
    """
    # print(f"Starting branching generation with {num_branches} branches")
    # print(f"Prompt: '{prompt}'")

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    # print(f"Tokenized prompt shape: {inputs['input_ids'].shape}")

    # Get initial top k tokens
    topk_values, topk_indices = get_topk_next_tokens(model, inputs, num_branches)
    
    # Log initial token choices
    for k in range(num_branches):
        token_text = tokenizer.decode(topk_indices[0, k])
        # print(f"Initial token {k+1}: '{token_text}' (prob: {topk_values[0,k]:.4f})")

    responses = []
    for k in range(num_branches):
        # print(f"\nGenerating branch {k+1}/{num_branches}")
        # Create a new branch starting with the k-th most likely token
        branch_inputs = {
            "input_ids": torch.cat(
                [inputs["input_ids"], topk_indices[:, k : k + 1]], dim=1
            ),
            "attention_mask": (
                torch.cat(
                    [
                        inputs["attention_mask"],
                        torch.ones((1, 1), device=inputs["attention_mask"].device),
                    ],
                    dim=1,
                )
                if "attention_mask" in inputs
                else None
            ),
        }

        # Generate the rest of the response for this branch
        generated_text, confidence_score = generate_single_branch(
            model, tokenizer, branch_inputs, max_length
        )

        responses.append((generated_text, confidence_score))
        # print(f"Branch {k+1} complete:")
        print(f"Generated text: '{generated_text}'")
        # print(f"Confidence score: {confidence_score:.4f}")

    print("\nAll branches complete")
    return responses