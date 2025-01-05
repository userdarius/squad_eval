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
    model: AutoModelForCausalLM, inputs: Dict[str, torch.Tensor], num_branches: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get the top k most likely next tokens and their probabilities.
    """
    with torch.no_grad():
        outputs = model(**inputs, return_dict=True, temperature=0.4)
        next_token_logits = outputs.logits[:, -1, :]

    probabilities = torch.softmax(next_token_logits, dim=-1)
    topk_values, topk_indices = torch.topk(probabilities, num_branches)

    return topk_values, topk_indices


def generate_single_branch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_length: int,
    inputs: Dict[str, torch.Tensor],
) -> Tuple[List[str], float]:
    # Initialize response_tokens with the first token from the input
    response_tokens = [
        inputs["input_ids"][0, -1].item()
    ]  # Get the last token from inputs
    prob_diffs = []

    print(f"Starting with initial token: '{tokenizer.decode([response_tokens[0]])}'")

    for step in range(max_length):
        topk_values, topk_indices = get_topk_next_tokens(model, inputs, num_branches=10)

        next_token = topk_indices[0, 0].item()
        next_token_text = tokenizer.decode([next_token])

        current_text = tokenizer.decode(response_tokens + [next_token])
        print(f"Step {step}: Token {next_token} -> '{next_token_text}'")
        print(f"Current text: '{current_text}'")

        # Check stopping conditions
        if any(
            stop in next_token_text for stop in [".", "\n", "Explanation:", "Answer:"]
        ):
            if next_token_text.strip() == "." and not any(
                stop in current_text for stop in [".", "\n", "Explanation:", "Answer:"]
            ):
                response_tokens.append(next_token)
            break

        # Regular token processing
        prob_diff = (topk_values[0, 0] - topk_values[0, 1]).item()
        response_tokens.append(next_token)
        prob_diffs.append(prob_diff)

        # Update inputs
        next_token_tensor = torch.tensor(
            [[next_token]], device=inputs["input_ids"].device
        )
        inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token_tensor], dim=1)
        if "attention_mask" in inputs:
            inputs["attention_mask"] = torch.cat(
                [
                    inputs["attention_mask"],
                    torch.ones((1, 1), device=inputs["attention_mask"].device),
                ],
                dim=1,
            )

    # Convert token IDs to text at the end
    generated_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
    avg_prob_diff = sum(prob_diffs) / len(prob_diffs) if prob_diffs else 0

    return generated_text.strip(), avg_prob_diff


def generate_branching_responses(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_length: int,
    num_branches: int,
) -> List[Tuple[str, float]]:
    """
    Generate multiple responses by exploring different initial tokens.
    """

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Get initial top k tokens
    topk_values, topk_indices = get_topk_next_tokens(model, inputs, num_branches)

    # Log initial token choices
    for k in range(num_branches):
        token_text = tokenizer.decode(topk_indices[0, k])
        print(f"Initial token {k+1}: {token_text} (prob: {topk_values[0,k]:.4f})")

    responses = []
    for k in range(num_branches):
        print(f"\nStarting branch {k+1}")
        first_token = topk_indices[:, k : k + 1]
        first_token_text = tokenizer.decode(first_token[0])
        print(f"First token: '{first_token_text}'")
        # if first token is a stop token, skip this branch
        if any(stop in first_token_text for stop in [".", "\n", "Explanation:", "Answer:"]):
            print(f"Skipping branch {k+1} because it starts with a stop token")
            continue
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
            model, tokenizer, max_length, branch_inputs
        )

        responses.append((generated_text, confidence_score))

    print("\nAll branches complete\n")
    return responses
