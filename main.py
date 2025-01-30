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
from typing import List, Tuple, Dict


def generate_answers(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    context: str,
    answer: str,
) -> Tuple[List[str], List[float], List[float]]:
    """
    Generate multiple answers for a given question and context using branching generation.

    Args:
        model: The language model
        tokenizer: The tokenizer
        question: The question to answer
        context: The context for the question
        num_branches: Number of different answers to generate

    Returns:
        Tuple of (list of answers, list of confidence scores, list of log probabilities)
    """
    prompt = f"Answer as simply as possible. Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    logging.info(f"Context: {context} \n\nQuestion: {question} \n\nAnswer: {answer}")

    # Use our branching generation method
    responses = generate_branching_responses(
        model,
        tokenizer,
        prompt,
        max_length=10,
        num_branches=10,
    )

    # Sort responses by confidence score
    responses.sort(key=lambda x: x[1], reverse=True)

    # Separate answers, confidence scores, and log probabilities
    answers = []
    confidence_scores = []
    log_probs = []

    prompt_end_index = len(prompt.strip())

    for (
        response,
        confidence_score,
        log_prob,
    ) in responses:
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
            confidence_scores.append(confidence_score)
            log_probs.append(log_prob)

            logging.info(f"Generated answer: {answer}")
            logging.info(f"Confidence score: {confidence_score}")
            logging.info(f"Log probability: {log_prob}")

    return answers, confidence_scores, log_probs


def normalize_answer(text):
    """Normalize answer text by removing articles, punctuation, and extra whitespace"""
    text = text.lower()
    text = "".join(char for char in text if char.isalnum() or char.isspace())
    text = " ".join(text.split())  # Normalize whitespace
    return text


def compute_exact_match(prediction, ground_truth):
    """Compute exact match after normalization"""
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def evaluate_sample(sample, model, tokenizer, entailment_model):
    """Evaluate semantic uncertainty metrics for a single sample."""
    answers, confidence_scores, log_probs = generate_answers(
        model,
        tokenizer,
        sample["question"],
        sample["context"],
        sample["answers"]["text"][0],
    )

    # Normalize answers
    answers = [normalize_answer(answer) for answer in answers]
    ground_truth = normalize_answer(sample["answers"]["text"][0])

    # Calculate exact match scores
    exact_matches = [compute_exact_match(answer, ground_truth) for answer in answers]
    exact_match_accuracy = sum(exact_matches) / len(exact_matches)

    # Calculate semantic IDs
    semantic_ids = get_semantic_ids(answers, entailment_model)

    # Basic metrics
    pred_entropy = predictive_entropy(np.array(log_probs))
    cluster_entropy = cluster_assignment_entropy(semantic_ids)
    context_entailment_score = context_entails_response(
        sample["context"], answers, entailment_model
    )
    answer_entailment_score = context_entails_response(
        sample["answers"]["text"][0], answers, entailment_model
    )

    # Print entailment scores
    print(f"Context entailment score: {context_entailment_score}")
    if context_entailment_score == 0:
        print(f"Contradiction")
    elif context_entailment_score == 1:
        print(f"Neutral")
    else:
        print(f"Entailment")

    print(f"Answer entailment score: {answer_entailment_score}")
    if answer_entailment_score == 0:
        print(f"Contradiction")
    elif answer_entailment_score == 1:
        print(f"Neutral")
    else:
        print(f"Entailment")

    # Calculate all metrics
    semantic_cluster_counts = np.bincount(semantic_ids)
    metrics = {
        # Basic Information
        "question_id": sample["id"],
        "question": sample["question"],
        "context": sample["context"],
        "generated_answers": answers,
        "ground_truth": sample["answers"]["text"][0],
        # Exact Match Metrics
        "exact_match_accuracy": float(exact_match_accuracy),
        "exact_matches": exact_matches,
        "exact_match_confidence_correlation": (
            float(np.corrcoef(exact_matches, confidence_scores)[0, 1])
            if len(exact_matches) > 1 and len(set(exact_matches)) > 1
            else 0.0
        ),
        # Basic Metrics
        "predictive_entropy": float(pred_entropy),
        "cluster_entropy": float(cluster_entropy),
        "context_entailment_score": float(context_entailment_score),
        "answer_entailment_score": float(answer_entailment_score),
        "semantic_clusters": semantic_ids.tolist(),
        "confidence_scores": confidence_scores,
        "log_probabilities": log_probs,
        # Sequence-Level Metrics
        "mean_sequence_length": float(np.mean([len(a.split()) for a in answers])),
        "response_diversity": float(len(set(answers)) / len(answers)),
        "max_logprob": float(max(log_probs)),
        "min_logprob": float(min(log_probs)),
        "logprob_range": float(max(log_probs) - min(log_probs)),
        # Cluster Analysis
        "num_semantic_clusters": int(len(set(semantic_ids))),
        "largest_cluster_size": int(max(semantic_cluster_counts)),
        "cluster_size_std": float(np.std(semantic_cluster_counts)),
        # Agreement Metrics
        "majority_answer_frequency": float(
            max(semantic_cluster_counts) / len(semantic_ids)
        ),
        "semantic_agreement_score": float(len(set(semantic_ids)) / len(answers)),
        # Correlation Analysis
        "logprob_confidence_correlation": float(
            np.corrcoef(log_probs, confidence_scores)[0, 1]
            if len(log_probs) > 1
            else 0.0
        ),
        "entropy_cluster_correlation": float(abs(pred_entropy - cluster_entropy)),
        # Consistency Metrics
        "context_answer_entailment_gap": float(
            abs(context_entailment_score - answer_entailment_score)
        ),
        # High Confidence Metrics
        "high_confidence_entailment": float(
            # Get the mean exact match score for the top 50% most confident answers
            np.mean(
                [
                    match
                    for conf, match in sorted(
                        zip(confidence_scores, exact_matches),
                        key=lambda x: x[0],  # Sort by confidence
                        reverse=True,  # Highest confidence first
                    )[
                        : len(confidence_scores) // 2
                    ]  # Take top half
                ]
            )
            if confidence_scores
            else 0.0
        ),
    }

    return metrics


def save_results(results: List[Dict], output_file: str) -> Dict:
    """Save evaluation results in a JSON file with mean results and individual samples."""
    df = pd.DataFrame(results)

    # Calculate mean and std for all numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    mean_results = {}

    for col in numeric_columns:
        if col == "question_id":
            continue
        if df[col].isna().all():
            logging.warning(f"Skipping column {col} with all NaN values")
            continue

        mean_val = df[col].mean()
        std_val = df[col].std()
        mean_results[col] = {
            "mean": float(mean_val) if not pd.isna(mean_val) else None,
            "std": float(std_val) if not pd.isna(std_val) else None,
        }

    # Create the combined JSON structure
    output_json = {"mean_results": mean_results, "samples": results}

    # Save to JSON file with UTF-8 encoding
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False)

    return mean_results


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

        # Shuffle the dataset
        shuffled_dataset = dataset.shuffle(seed=42)
        logging.info(
            f"Dataset shuffled successfully with {len(shuffled_dataset)} samples"
        )

        # Initialize results storage
        results = []

        # Process samples
        for idx, sample in enumerate(tqdm(shuffled_dataset)):
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
        output_file = f"semantic_uncertainty_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        metrics = save_results(results, output_file)
        logging.info(f"Results saved to {output_file}")
        logging.info(f"Summary metrics: {metrics}")

    except Exception as e:
        logging.critical(f"Critical error in main execution: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
