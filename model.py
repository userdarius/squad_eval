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


### Chain of Thought ###
def get_topk_tokens_with_logits(model, inputs, num_branches=10):
    """Get top k tokens, their probabilities, and full logits for logging."""
    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)
        next_token_logits = outputs.logits[:, -1, :]
    
    # Get full logits for log probability calculation
    logits = outputs.logits
    
    # Apply softmax to convert logits to probabilities
    probabilities = torch.softmax(next_token_logits, dim=-1)

    # Get the top k tokens and their probabilities
    topk_values, topk_indices = torch.topk(probabilities, num_branches)

    return topk_values, topk_indices, logits

def generate_response_with_scores(model, tokenizer, inputs, max_length=500):
    """Generate a response while tracking both probability differences and log probabilities."""
    response = []
    response_probs = []  # For confidence scoring
    all_logits = []  # For log probability calculation
    
    for i in range(max_length):
        # Generate the logits for the next token
        topk_values, topk_indices, logits = get_topk_tokens_with_logits(model, inputs, num_branches=2)
        all_logits.append(logits)

        # Get the difference in probabilities between the top two tokens
        prob_diff = topk_values[:, 0] - topk_values[:, 1]
        response_probs.append(prob_diff.item())

        # Append the most likely token to the response
        next_token = topk_indices[:, 0]
        response.append(next_token)

        if next_token == tokenizer.eos_token_id:
            break

        # Add the token to the input for the next iteration
        inputs['input_ids'] = torch.cat([inputs['input_ids'], next_token.unsqueeze(-1)], dim=1)

    response_tensor = torch.cat(response).unsqueeze(0)
    return response_tensor, response_probs, all_logits

def generate_branching_responses(model, tokenizer, prompt, num_branches=10, max_length=500):
    """Generate multiple responses with both branching and score tracking."""
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Get initial top k tokens
    topk_values, topk_indices, initial_logits = get_topk_tokens_with_logits(model, inputs, num_branches)

    responses = []
    response_probs = []  # Confidence scores
    all_branch_logits = []  # For log probability calculation
    
    for k in range(num_branches):
        # Create new branch with kth most likely token
        new_input_ids = inputs.copy()
        new_input_ids['input_ids'] = torch.cat([inputs['input_ids'], topk_indices[:, k].unsqueeze(-1)], dim=1)

        # Generate full response for this branch
        response, probs, logits = generate_response_with_scores(model, tokenizer, new_input_ids, max_length)
        
        responses.append(response)
        response_probs.append(sum(probs) / len(probs))
        all_branch_logits.append([initial_logits] + logits)

    # Create a structure similar to the original model outputs
    class OutputsWithScores:
        def __init__(self, sequences, scores, logits):
            self.sequences = sequences
            self.scores = scores
            self.logits = logits

    outputs = OutputsWithScores(
        sequences=torch.cat(responses, dim=0),
        scores=[logits[-1] for logits in all_branch_logits],
        logits=all_branch_logits
    )

    return outputs, response_probs

