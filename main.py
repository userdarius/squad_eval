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
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from model import (
    load_model_and_tokenizer,
    EntailmentDeberta,
    generate_branching_responses,
)
from data import get_dataset
from scores import (
    context_entails_response,
    get_semantic_ids,
    predictive_entropy,
    cluster_assignment_entropy,
)
from typing import List, Tuple


def generate_answers(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    context: str,
    answer: str,
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
    prompt = f"Answer as simply as possible. Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    logging.info(f"Context: {context} \n\nQuestion: {question} \n\nAnswer: {answer}")

    # Use our branching generation method
    responses = generate_branching_responses(
        model,
        tokenizer,
        prompt,
        max_length=5,
        num_branches=10,
    )

    # Sort responses by confidence score
    responses.sort(key=lambda x: x[1], reverse=True)

    # Separate answers and probabilities
    answers = []
    log_probs = []

    prompt_end_index = len(prompt.strip())

    for response, prob in responses:
        # Extract just the answer part, making sure to handle the full response correctly
        full_response = response.strip()

        # If the response starts with the prompt (or part of it), remove it
        if full_response.startswith(prompt):  
            answer = full_response[prompt_end_index:].strip()
        else:
            # If response doesn't include prompt, use it as is
            answer = full_response.strip()

        # Only add non-empty answers
        if answer:
            answers.append(answer)
            log_probs.append(prob)

            logging.info(f"Generated answer: {answer}")
            logging.info(f"Log probability: {prob}")

    return answers, log_probs


def evaluate_sample(sample, model, tokenizer, entailment_model):
    """Evaluate semantic uncertainty metrics for a single sample."""
    answers, log_probs = generate_answers(
        model,
        tokenizer,
        sample["question"],
        sample["context"],
        sample["answers"]["text"][0],
    )

    # Calculate semantic IDs
    semantic_ids = get_semantic_ids(answers, entailment_model)

    # Calculate metrics
    pred_entropy = predictive_entropy(np.array(log_probs))
    cluster_entropy = cluster_assignment_entropy(semantic_ids)
    context_entailment_score = context_entails_response(
        sample["context"], answers, entailment_model
    )
    print(f"Context entailment score: {context_entailment_score}")
    if context_entailment_score == 0:
        print(f"Contradiction")
    elif context_entailment_score == 1:
        print(f"Neutral")
    else:
        print(f"Entailment")

    answer_entailment_score = context_entails_response(
        sample["answers"]["text"][0], answers, entailment_model
    )
    print(f"Answer entailment score: {answer_entailment_score}")
    if answer_entailment_score == 0:
        print(f"Contradiction")
    elif answer_entailment_score == 1:
        print(f"Neutral")
    else:
        print(f"Entailment")

    return {
        "question_id": sample["id"],
        "question": sample["question"],
        "context": sample["context"],
        "generated_answers": answers,
        "ground_truth": sample["answers"]["text"][0],
        "predictive_entropy": pred_entropy,
        "cluster_entropy": cluster_entropy,
        "context_entailment_score": context_entailment_score,
        "answer_entailment_score": answer_entailment_score,
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
        "mean_context_entailment_score": df["context_entailment_score"].mean(),
        "mean_answer_entailment_score": df["answer_entailment_score"].mean(),
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
            if idx >= 5:  # Limit to 100 samples for testing
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
