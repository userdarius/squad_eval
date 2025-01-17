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
        answer = generated_text[len(prompt):].strip()

        # Post-process answer
        for stop_token in stopping_tokens:
            if stop_token in answer:
                answer = answer[:answer.index(stop_token) + len(stop_token)]
                break

        # Calculate sequence log probability and confidence score
        scores = torch.stack(outputs.scores, dim=1)
        sequence = outputs.sequences[0, inputs["input_ids"].size(1):]  # Only get new tokens
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
            avg_confidence = sum(confidence_per_token) / len(confidence_per_token)  # Keep confidence normalized

            answers.append(answer)
            log_probs.append(sequence_log_prob)
            confidence_scores.append(avg_confidence)

            logging.info(f"Generated answer: {answer}")
            logging.info(f"Raw log probability: {sequence_log_prob}")  # Updated log message
            logging.info(f"Confidence score: {avg_confidence}")

    return answers, confidence_scores, log_probs


def normalize_answer(text):
    """Normalize answer text by removing periods and extra whitespace"""
    return text.rstrip(".")


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

    # Calculate semantic IDs
    semantic_ids = get_semantic_ids(answers, entailment_model)

    # Calculate metrics
    pred_entropy = predictive_entropy(np.array(log_probs))
    print(f"Predictive entropy: {pred_entropy}")
    cluster_entropy = cluster_assignment_entropy(semantic_ids)
    print(f"Cluster assignment entropy: {cluster_entropy}")
    context_entailment_score = context_entails_response(
        sample["context"], answers, entailment_model
    )
    answer_entailment_score = context_entails_response(
        sample["answers"]["text"][0], answers, entailment_model
    )

    # Print entailment scores (existing logic)
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

    semantic_cluster_counts = np.bincount(semantic_ids)

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
        "confidence_scores": confidence_scores,
        "log_probabilities": log_probs,
        "mean_sequence_length": np.mean([len(a.split()) for a in answers]),
        "response_diversity": len(set(answers)) / len(answers),
        "max_logprob": max(log_probs),
        "min_logprob": min(log_probs),
        "logprob_range": max(log_probs) - min(log_probs),
        "num_semantic_clusters": len(set(semantic_ids)),
        "largest_cluster_size": max(semantic_cluster_counts),
        "cluster_size_std": np.std(semantic_cluster_counts),
        "majority_answer_frequency": max(semantic_cluster_counts) / len(semantic_ids),
        "semantic_agreement_score": len(set(semantic_ids)) / len(answers),
        "logprob_confidence_correlation": np.corrcoef(log_probs, confidence_scores)[
            0, 1
        ],
        "entropy_cluster_correlation": abs(pred_entropy - cluster_entropy),
        "context_answer_entailment_gap": abs(
            context_entailment_score - answer_entailment_score
        ),
        "high_confidence_entailment": np.mean(
            [c for c, s in zip(confidence_scores, semantic_ids) if s == semantic_ids[0]]
        ),
    }


def save_results(results: List[Dict], output_file: str) -> Dict:
    """Save evaluation results and create visualizations."""
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)

    # Calculate summary statistics for all numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    metrics = {
        f"mean_{col}": df[col].mean()
        for col in numeric_columns
        if col not in ["question_id"] and not df[col].isna().all()
    }

    # Add standard deviations
    metrics.update(
        {
            f"std_{col}": df[col].std()
            for col in numeric_columns
            if col not in ["question_id"] and not df[col].isna().all()
        }
    )

    # Save metrics
    with open(output_file.replace(".csv", "_summary.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Create visualizations
    output_prefix = output_file.replace(".csv", "")
    create_visualizations(df, output_prefix)

    return metrics


def create_visualizations(df: pd.DataFrame, output_prefix: str):
    """Create and save various visualizations from the results DataFrame."""

    # Set up the style
    plt.style.use("seaborn")

    # 1. Correlation Heatmap of Numeric Metrics
    plt.figure(figsize=(12, 10))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0)
    plt.title("Correlation Between Uncertainty Metrics")
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_correlation_heatmap.png")
    plt.close()

    # 2. Joint Plot of Entropies
    plt.figure(figsize=(10, 6))
    sns.jointplot(
        data=df,
        x="predictive_entropy",
        y="cluster_entropy",
        kind="scatter",
        joint_kws={"alpha": 0.5},
    )
    plt.suptitle("Relationship Between Predictive and Cluster Entropy")
    plt.savefig(f"{output_prefix}_entropy_relationship.png")
    plt.close()

    # 3. Distribution of Semantic Clusters
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, y="num_semantic_clusters")
    plt.title("Distribution of Number of Semantic Clusters")
    plt.savefig(f"{output_prefix}_semantic_clusters_dist.png")
    plt.close()

    # 4. Confidence vs Agreement Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df, x="high_confidence_entailment", y="semantic_agreement_score", alpha=0.6
    )
    plt.title("Confidence vs Semantic Agreement")
    plt.xlabel("High Confidence Entailment")
    plt.ylabel("Semantic Agreement Score")
    plt.savefig(f"{output_prefix}_confidence_agreement.png")
    plt.close()

    # Entailment-based metrics
    plt.figure(figsize=(12, 8))

    metrics = [
        "context_answer_entailment_gap",
        "high_confidence_entailment",
        "entropy_cluster_correlation",
    ]

    plt.figure(figsize=(15, 5))
    for idx, metric in enumerate(metrics, 1):
        plt.subplot(1, 3, idx)
        sns.kdeplot(data=df[metric], fill=True)
        plt.title(f"{metric} Distribution")
        plt.xlabel(metric)

    plt.tight_layout()
    plt.savefig("entailment_analysis.png")
    plt.close()

    # Sequence-based metrics
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Sequence Metrics Analysis")

    sns.boxplot(data=df["mean_sequence_length"], ax=axes[0])
    axes[0].set_title("Mean Sequence Length Distribution")
    axes[0].set_xlabel("Mean Sequence Length")

    sns.histplot(data=df["response_diversity"], ax=axes[1], kde=True)
    axes[1].set_title("Response Diversity Distribution")
    axes[1].set_xlabel("Response Diversity")

    plt.tight_layout()
    plt.savefig("sequence_metrics_analysis.png")
    plt.close()

    # 5. Multiple Metric Comparison
    metrics_to_compare = [
        "predictive_entropy",
        "cluster_entropy",
        "context_answer_entailment_gap",
        "response_diversity",
    ]
    plt.figure(figsize=(12, 6))
    df[metrics_to_compare].boxplot()
    plt.xticks(rotation=45)
    plt.title("Distribution of Key Uncertainty Metrics")
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_metrics_distribution.png")
    plt.close()

    # Select key metrics from each category
    key_metrics = {
        "Semantic": ["semantic_agreement_score", "num_semantic_clusters"],
        "Probability": ["max_logprob", "logprob_range"],
        "Entailment": [
            "context_answer_entailment_gap",
            "high_confidence_entailment",
        ],
        "Sequence": ["mean_sequence_length", "response_diversity"],
    }

    # Create correlation matrix for these metrics
    selected_metrics = [m for metrics in key_metrics.values() for m in metrics]
    correlation_matrix = df[selected_metrics].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0)
    plt.title("Cross-Category Metric Correlations")
    plt.tight_layout()
    plt.savefig("comprehensive_metric_relationships.png")
    plt.close()

    # 6. Sequence Length vs Log Probability
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="mean_sequence_length", y="max_logprob", alpha=0.6)
    plt.title("Sequence Length vs Maximum Log Probability")
    plt.savefig(f"{output_prefix}_length_vs_probability.png")
    plt.close()



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
        output_file = f"semantic_uncertainty_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        metrics = save_results(results, output_file)
        logging.info(f"Results saved to {output_file}")
        logging.info(f"Summary metrics: {metrics}")

    except Exception as e:
        logging.critical(f"Critical error in main execution: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
