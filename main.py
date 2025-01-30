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
from typing import List, Dict
import torch.nn.functional as F


def generate_answers(model, tokenizer, question, context, answer, num_samples=10):
    """Generate multiple answers for a given question and context."""
    prompt = f"Answer as simply as possible. Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    logging.info(f"Context: {context} \n\nQuestion: {question} \n\nAnswer: {answer}")

    stopping_tokens = [".", "\n\n", "\n"]
    answers = []
    log_probs = []
    confidence_scores = []

    for _ in range(num_samples):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                num_return_sequences=1,
                output_scores=True,
                return_dict_in_generate=True,
                no_repeat_ngram_size=3,
                length_penalty=1.2,
                temperature=0.4,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Get generated text
        generated_sequence = outputs.sequences[0]
        generated_text = tokenizer.decode(generated_sequence, skip_special_tokens=True)
        answer = generated_text[len(prompt) :].strip()

        # Post-process answer
        for stop_token in stopping_tokens:
            if stop_token in answer:
                answer = answer[: answer.index(stop_token) + len(stop_token)]
                break

        # Calculate sequence log probability and confidence score
        scores = torch.stack(outputs.scores, dim=1)
        sequence = outputs.sequences[
            0, inputs["input_ids"].size(1) :
        ]  # Only get new tokens
        seq_length = sequence.size(0)

        if seq_length > 0:
            log_probs_per_token = []
            confidence_per_token = []

            for i in range(min(seq_length, scores.size(1))):
                # Calculate token probabilities
                token_probs = F.softmax(scores[0, i], dim=-1)
                top_probs, _ = torch.topk(token_probs, k=2)

                # Calculate confidence as prob difference (like branching)
                token_confidence = (top_probs[0] - top_probs[1]).item()
                confidence_per_token.append(token_confidence)

                # Calculate log probability
                token_log_probs = F.log_softmax(scores[0, i], dim=-1)
                token_log_prob = token_log_probs[sequence[i]].item()
                log_probs_per_token.append(token_log_prob)

            # Use raw sum instead of normalized average
            sequence_log_prob = sum(log_probs_per_token)  # Removed normalization
            avg_confidence = sum(confidence_per_token) / len(
                confidence_per_token
            )  # Keep confidence normalized

            answers.append(answer)
            log_probs.append(sequence_log_prob)
            confidence_scores.append(avg_confidence)

            logging.info(f"Generated answer: {answer}")
            logging.info(
                f"Raw log probability: {sequence_log_prob}"
            )  # Updated log message
            logging.info(f"Confidence score: {avg_confidence}")

    return answers, confidence_scores, log_probs


def normalize_answer(text):
    """Normalize answer text by removing periods and extra whitespace"""
    return text.rstrip(".")


def normalize_answer(text):
    """Normalize answer text by removing articles, punctuation, and extra whitespace"""
    text = text.lower()
    text = ''.join(char for char in text if char.isalnum() or char.isspace())
    text = ' '.join(text.split())  # Normalize whitespace
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

    # Calculate metrics
    pred_entropy = predictive_entropy(np.array(log_probs))
    cluster_entropy = cluster_assignment_entropy(semantic_ids)
    context_entailment_score = context_entails_response(
        sample["context"], answers, entailment_model
    )
    answer_entailment_score = context_entails_response(
        sample["answers"]["text"][0], answers, entailment_model
    )

    semantic_cluster_counts = np.bincount(semantic_ids)

    return {
        "question_id": sample["id"],
        "question": sample["question"],
        "context": sample["context"],
        "generated_answers": answers,
        "ground_truth": sample["answers"]["text"][0],
        "exact_match_accuracy": float(exact_match_accuracy),
        "exact_matches": exact_matches,
        "predictive_entropy": float(pred_entropy),
        "cluster_entropy": float(cluster_entropy),
        "context_entailment_score": float(context_entailment_score),
        "answer_entailment_score": float(answer_entailment_score),
        "semantic_clusters": semantic_ids,
        "confidence_scores": confidence_scores,
        "log_probabilities": log_probs,
        "mean_sequence_length": float(np.mean([len(a.split()) for a in answers])),
        "response_diversity": len(set(answers)) / len(answers),
        "max_logprob": max(log_probs) if log_probs else 0.0,
        "min_logprob": min(log_probs) if log_probs else 0.0,
        "logprob_range": (max(log_probs) - min(log_probs)) if log_probs else 0.0,
        "num_semantic_clusters": len(set(semantic_ids)),
        "largest_cluster_size": int(max(semantic_cluster_counts)),
        "cluster_size_std": float(np.std(semantic_cluster_counts)),
        "majority_answer_frequency": float(
            max(semantic_cluster_counts) / len(semantic_ids)
        ),
        "semantic_agreement_score": len(set(semantic_ids)) / len(answers),
        "logprob_confidence_correlation": (
            float(np.corrcoef(log_probs, confidence_scores)[0, 1])
            if len(log_probs) > 1
            else 0.0
        ),
        "entropy_cluster_correlation": float(abs(pred_entropy - cluster_entropy)),
        "context_answer_entailment_gap": float(
            abs(context_entailment_score - answer_entailment_score)
        ),
        "high_confidence_entailment": (
            float(
                # Get the mean exact match score for the top 50% most confident answers
                np.mean([
                    match for conf, match in sorted(
                        zip(confidence_scores, exact_matches),
                        key=lambda x: x[0],  # Sort by confidence
                        reverse=True  # Highest confidence first
                    )[:len(confidence_scores)//2]  # Take top half
                ])
            )
            if confidence_scores
            else 0.0
        ),
        "exact_match_confidence_correlation": (
            float(np.corrcoef(exact_matches, confidence_scores)[0, 1])
            if len(exact_matches) > 1 and len(set(exact_matches)) > 1
            else 0.0
        ),
    }


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
            "std": float(std_val) if not pd.isna(std_val) else None
        }
    
    # Create the combined JSON structure
    output_json = {
        "mean_results": mean_results,
        "samples": results
    }
    
    # Save to JSON file with UTF-8 encoding
    with open(output_file, "w", encoding="utf-8") as f:  # Explicit encoding
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

        # Initialize results storage
        results = []

        # Process samples
        for idx, sample in enumerate(tqdm(shuffled_dataset)):
            if idx >= 500:  # Limit to 10 samples for testing
                break

            try:
                result = evaluate_sample(sample, model, tokenizer, entailment_model)
                results.append(result)

                if (idx + 1) % 10 == 0:
                    logging.info(f"Processed {idx + 1} samples")

                torch.cuda.empty_cache()
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
