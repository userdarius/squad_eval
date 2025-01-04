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
    generate_branching_responses,
)
from data import get_dataset
from scores import (
    context_entails_response,
    get_semantic_ids,
    predictive_entropy,
    cluster_assignment_entropy,
)


def generate_answers(model, tokenizer, question, context, num_samples=5):
    """Generate multiple answers using branching and tracking both types of scores."""
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"

    answers = []
    log_probs = []
    confidence_scores = []

    logging.info(f"Context: {context}")
    logging.info(f"Question: {question}")

    for _ in range(num_samples):
        try:
            # Generate branching responses with longer max_length and temperature
            outputs, branch_scores = generate_branching_responses(
                model,
                tokenizer,
                prompt,
                num_branches=10,
                max_length=50,  # Increased from 20
            )

            # Get the most confident response from branches
            best_branch_idx = torch.argmax(torch.tensor(branch_scores))
            
            # Get the full sequence for the best branch
            generated_sequence = outputs.sequences[best_branch_idx]
            
            # Find where the prompt ends in the generated sequence
            prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
            prompt_length = len(prompt_ids)
            
            # Extract only the answer part
            answer_ids = generated_sequence[prompt_length:]
            
            # Decode the answer
            answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
            
            # Only append if we got a non-empty answer
            if answer:
                answers.append(answer)
                confidence_scores.append(branch_scores[best_branch_idx])

                # Calculate sequence log probability for the best response
                sequence_logits = outputs.logits[best_branch_idx]
                
                # Get the minimum sequence length
                min_seq_length = min(logits.size(1) for logits in sequence_logits)
                
                # Truncate all logits to the minimum length
                truncated_logits = [logits[:, :min_seq_length, :] for logits in sequence_logits]
                
                # Stack the tensors
                sequence_scores = torch.stack(truncated_logits, dim=0)
                
                # Calculate log probability with proper handling
                log_probs_per_token = torch.log_softmax(sequence_scores[:, 0, :], dim=-1)
                sequence_log_prob = torch.sum(torch.max(log_probs_per_token, dim=-1)[0])
                log_probs.append(float(sequence_log_prob))
            else:
                logging.warning("Generated an empty response, retrying...")
                continue

        except Exception as e:
            logging.error(f"Error in generation: {str(e)}")
            continue

    # If we didn't get enough valid responses, fill with defaults
    while len(answers) < num_samples:
        answers.append("Unable to generate response")
        log_probs.append(float('-inf'))
        confidence_scores.append(0.0)

    return answers, log_probs, confidence_scores


def evaluate_sample(sample, model, tokenizer, entailment_model):
    """Evaluate semantic uncertainty metrics for a single sample."""
    answers, log_probs, confidence_scores = generate_answers(
        model, tokenizer, sample["question"], sample["context"]
    )

    # Print results with both types of scores
    for i, (answer, log_prob, confidence) in enumerate(zip(answers, log_probs, confidence_scores)):
        print(f"\nAnswer {i+1}:")
        print(f"Text: {answer}")
        print(f"Log Probability: {log_prob}")
        print(f"Confidence Score: {confidence}")

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
