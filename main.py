"""
Main script to run the semantic entropy evaluation on squad dataset with enhanced logging
"""

import sys
from datetime import datetime
import logging
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from model import (
    load_model_and_tokenizer,
    EntailmentDeberta,
)
from data import get_dataset
from scores import (
    context_entails_response,
    get_semantic_ids,
    predictive_entropy,
    cluster_assignment_entropy,
)


import torch
import torch.nn.functional as F
import logging
from typing import List, Tuple, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_topk_next_tokens(
    model: AutoModelForCausalLM, inputs: Dict[str, torch.Tensor], num_branches: int = 5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get the top k most likely next tokens and their probabilities.

    Args:
        model: The language model
        inputs: Tokenized inputs
        num_branches: Number of top tokens to return

    Returns:
        Tuple of (probabilities, token indices)
    """
    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)
        next_token_logits = outputs.logits[:, -1, :]

    # Get probabilities and top k tokens
    probabilities = F.softmax(next_token_logits, dim=-1)
    topk_values, topk_indices = torch.topk(probabilities, num_branches)

    return topk_values, topk_indices


def generate_single_branch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    inputs: Dict[str, torch.Tensor],
    max_length: int = 100,
) -> Tuple[List[str], float]:
    """
    Generate a complete response starting from the given inputs.

    Args:
        model: The language model
        tokenizer: The tokenizer
        inputs: Initial tokenized inputs
        max_length: Maximum generation length

    Returns:
        Tuple of (generated tokens, average probability difference)
    """
    response = []
    prob_diffs = []

    for _ in range(max_length):
        # Get top 2 most likely tokens to calculate probability difference
        topk_values, topk_indices = get_topk_next_tokens(model, inputs, num_branches=2)

        # Calculate probability difference between top two tokens
        prob_diff = (topk_values[0, 0] - topk_values[0, 1]).item()
        prob_diffs.append(prob_diff)

        # Add most likely token to response
        next_token = topk_indices[0, 0].unsqueeze(0)
        response.append(next_token)

        # Stop if we hit the end token
        if next_token.item() == tokenizer.eos_token_id:
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
    generated_text = tokenizer.decode(torch.cat(response))
    avg_prob_diff = sum(prob_diffs) / len(prob_diffs) if prob_diffs else 0

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

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt
        num_branches: Number of different branches to explore
        max_length: Maximum generation length

    Returns:
        List of tuples containing (generated text, confidence score)
    """
    logging.info(f"Generating {num_branches} branching responses for prompt: {prompt}")

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Get initial top k tokens
    topk_values, topk_indices = get_topk_next_tokens(model, inputs, num_branches)

    responses = []
    for k in range(num_branches):
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
        logging.info(
            f"Generated branch {k+1}/{num_branches} with confidence: {confidence_score:.4f}"
        )

    return responses


# Example usage
def generate_answers(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    context: str,
    num_branches: int = 5,
) -> Tuple[List[str], List[float]]:
    """
    Generate multiple answers for a given question and context using branching generation.

    Args:
        model: The language model
        tokenizer: The tokenizer
        question: The question to answer
        context: The context for the question
        num_branches: Number of different answers to generate

    Returns:
        Tuple of (list of answers, list of log probabilities)
    """
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    logging.info(f"Context: {context}")
    logging.info(f"Question: {question}")

    # Use our branching generation method
    responses = generate_branching_responses(
        model,
        tokenizer,
        prompt,
        num_branches=num_branches,
        max_length=20,  # Using same max_length as original
    )

    # Sort responses by confidence score
    responses.sort(key=lambda x: x[1], reverse=True)

    # Separate answers and probabilities and remove the prompt from answers
    answers = []
    log_probs = []

    for response, prob in responses:
        # Extract just the answer part after the prompt
        answer = response[len(prompt) :].strip()
        answers.append(answer)
        log_probs.append(prob)

        logging.info(f"Generated answer: {answer}")
        logging.info(f"Log probability: {prob}")

    return answers, log_probs


def evaluate_sample(sample, model, tokenizer, entailment_model):
    """Evaluate semantic uncertainty metrics for a single sample."""
    answers, log_probs = generate_answers(
        model, tokenizer, sample["question"], sample["context"]
    )

    # Calculate semantic IDs
    semantic_ids = get_semantic_ids(answers, entailment_model)

    # Calculate metrics
    pred_entropy = predictive_entropy(np.array(log_probs))
    cluster_entropy = cluster_assignment_entropy(semantic_ids)
    entailment_score = context_entails_response(
        sample["context"], answers, entailment_model
    )

    return {
        "question_id": sample["id"],
        "question": sample["question"],
        "context": sample["context"],
        "generated_answers": answers,
        "ground_truth": sample["answers"]["text"][0],
        "predictive_entropy": pred_entropy,
        "cluster_entropy": cluster_entropy,
        "entailment_score": entailment_score,
        "semantic_clusters": semantic_ids,
    }


def save_results(results, output_file):
    """Save evaluation results to a file."""
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)

    # Calculate and log summary statistics
    metrics = {
        "mean_predictive_entropy": df["predictive_entropy"].mean(),
        "mean_cluster_entropy": df["cluster_entropy"].mean(),
        "mean_entailment_score": df["entailment_score"].mean(),
    }

    with open(output_file.replace(".csv", "_summary.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(
                f'semantic_evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
            ),
            logging.StreamHandler(),
        ],
    )

    logging.info("Starting semantic entropy evaluation")

    try:
        # Load models and tokenizer
        logging.info("Loading models and tokenizer")
        model_name = "meta-llama/Llama-3.1-8B-Instruct"
        model, tokenizer = load_model_and_tokenizer(model_name)
        entailment_model = EntailmentDeberta()
        logging.info("Models loaded successfully")

        # Load dataset
        logging.info("Loading squad dataset")
        dataset = get_dataset("squad")[
            "validation"
        ]  # Using validation set for evaluation
        logging.info(f"Dataset loaded successfully with {len(dataset)} samples")

        # Initialize results storage
        results = []

        # Process samples
        for idx, sample in enumerate(tqdm(dataset)):
            if idx >= 100:  # Limit to 100 samples for testing
                break

            try:
                result = evaluate_sample(sample, model, tokenizer, entailment_model)
                results.append(result)

                if (idx + 1) % 10 == 0:
                    logging.info(f"Processed {idx + 1} samples")

            except Exception as e:
                logging.error(f"Error processing sample {idx}: {str(e)}", exc_info=True)
                continue

        # Save results
        output_file = f"semantic_uncertainty_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        metrics = save_results(results, output_file)
        logging.info(f"Results saved to {output_file}")
        logging.info(f"Summary metrics: {metrics}")

    except Exception as e:
        logging.critical(f"Critical error in main execution: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
