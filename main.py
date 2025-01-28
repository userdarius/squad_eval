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
    load_approx_and_target_model_and_tokenizer,
    speculative_sampling,
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


def generate_answers_with_spec_sampling(
    approx_model,
    target_model,
    tokenizer,
    question,
    context,
    answer,
    num_samples=10,
    max_new_tokens=10,
    gamma=4,
    temperature=0.4,
):
    """Generate multiple answers using speculative sampling."""
    prompt = f"Answer as simply as possible. Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    logging.info(f"Context: {context} \n\nQuestion: {question} \n\nAnswer: {answer}")

    stopping_tokens = [".", "\n\n", "\n"]
    answers = []
    log_probs = []
    confidence_scores = []

    # Encode the prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(target_model.device)

    for _ in range(num_samples):
        # Generate sequence using speculative sampling
        with torch.no_grad():
            output_sequence, sequence_log_probs = speculative_sampling(
                prefix=input_ids,
                approx_model=approx_model,
                target_model=target_model,
                max_len=max_new_tokens,
                gamma=gamma,
                temperature=temperature,
                top_p=0.9,
                verbose=False,
            )

        # Decode generated sequence
        generated_text = tokenizer.decode(output_sequence[0], skip_special_tokens=True)
        answer = generated_text[len(prompt) :].strip()

        # Post-process answer
        for stop_token in stopping_tokens:
            if stop_token in answer:
                answer = answer[: answer.index(stop_token) + len(stop_token)]
                break

        # Calculate sequence-level metrics
        if len(sequence_log_probs) > 0:
            # Sum log probabilities for the sequence
            sequence_log_prob = sum(sequence_log_probs)

            # Calculate confidence as difference between top token probabilities
            confidence_scores_per_token = []
            for log_prob in sequence_log_probs:
                # Convert log prob to probability
                prob = torch.exp(torch.tensor(log_prob))
                # Use a simple confidence metric based on probability
                confidence = prob.item()
                confidence_scores_per_token.append(confidence)

            avg_confidence = sum(confidence_scores_per_token) / len(
                confidence_scores_per_token
            )

            answers.append(answer)
            log_probs.append(sequence_log_prob)
            confidence_scores.append(avg_confidence)

            logging.info(f"Generated answer: {answer}")
            logging.info(f"Raw log probability: {sequence_log_prob}")
            logging.info(f"Confidence score: {avg_confidence}")

    return answers, confidence_scores, log_probs


def normalize_answer(text):
    """Normalize answer text by removing periods and extra whitespace"""
    return text.rstrip(".")


def evaluate_sample(sample, approx_model, target_model, tokenizer, entailment_model):
    """Evaluate semantic uncertainty metrics for a single sample using speculative sampling."""
    answers, confidence_scores, log_probs = generate_answers_with_spec_sampling(
        approx_model,
        target_model,
        tokenizer,
        sample["question"],
        sample["context"],
        sample["answers"]["text"][0],
    )

    # Rest of the evaluation remains the same
    answers = [normalize_answer(answer) for answer in answers]
    semantic_ids = get_semantic_ids(answers, entailment_model)

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

    logging.info("Starting semantic entropy evaluation with speculative sampling")

    try:
        # Load models and tokenizer
        logging.info("Loading models and tokenizer")
        target_model_name = "meta-llama/Llama-3.2-3B-Instruct"
        approx_model_name = "meta-llama/Llama-3.2-1B-Instruct"

        approx_model, target_model, tokenizer = (
            load_approx_and_target_model_and_tokenizer(
                approx_model_name, target_model_name
            )
        )
        entailment_model = EntailmentDeberta()
        logging.info("Models loaded successfully")

        # Load dataset
        logging.info("Loading squad dataset")
        dataset = get_dataset("squad")["validation"]
        logging.info(f"Dataset loaded successfully with {len(dataset)} samples")

        # Shuffle the dataset
        shuffled_dataset = dataset.shuffle(seed=42)

        # Initialize results storage
        results = []

        # Process samples
        for idx, sample in enumerate(tqdm(shuffled_dataset)):
            if idx >= 500:  # Limit to 500 samples
                break

            try:
                result = evaluate_sample(
                    sample, approx_model, target_model, tokenizer, entailment_model
                )
                results.append(result)

                if (idx + 1) % 10 == 0:
                    logging.info(f"Processed {idx + 1} samples")

                torch.cuda.empty_cache()
            except Exception as e:
                logging.error(f"Error processing sample {idx}: {str(e)}", exc_info=True)
                continue

        # Save results
        output_file = f"semantic_uncertainty_results_spec_sampling_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        metrics = save_results(results, output_file)
        logging.info(f"Results saved to {output_file}")
        logging.info(f"Summary metrics: {metrics}")

    except Exception as e:
        logging.critical(f"Critical error in main execution: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
