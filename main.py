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

def generate_answers(model, tokenizer, question, context, num_samples=5):
    """Generate multiple answers for a given question and context."""
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    
    answers = []
    log_probs = []
    
    for _ in range(num_samples):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                num_return_sequences=1,
                output_scores=True,
                return_dict_in_generate=True,
                no_repeat_ngram_size=3,
                length_penalty=1.0,
                temperature=0.5,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
            
        generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        answer = generated_text[len(prompt):].strip()
        answers.append(answer)
        
        # Calculate sequence log probability
        sequence_scores = torch.stack(outputs.scores, dim=1)
        log_prob = torch.sum(torch.log_softmax(sequence_scores[0], dim=-1))
        log_probs.append(log_prob.item())
    
    return answers, log_probs

def evaluate_sample(sample, model, tokenizer, entailment_model):
    """Evaluate semantic uncertainty metrics for a single sample."""
    answers, log_probs = generate_answers(
        model, 
        tokenizer, 
        sample["question"], 
        sample["context"]
    )
    
    # Calculate semantic IDs
    semantic_ids = get_semantic_ids(answers, entailment_model)
    
    # Calculate metrics
    pred_entropy = predictive_entropy(np.array(log_probs))
    cluster_entropy = cluster_assignment_entropy(semantic_ids)
    entailment_score = context_entails_response(sample["context"], answers, entailment_model)
    
    return {
        "question_id": sample["id"],
        "question": sample["question"],
        "context": sample["context"],
        "generated_answers": answers,
        "ground_truth": sample["answers"]["text"][0],
        "predictive_entropy": pred_entropy,
        "cluster_entropy": cluster_entropy,
        "entailment_score": entailment_score,
        "semantic_clusters": semantic_ids
    }

def save_results(results, output_file):
    """Save evaluation results to a file."""
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    
    # Calculate and log summary statistics
    metrics = {
        "mean_predictive_entropy": df["predictive_entropy"].mean(),
        "mean_cluster_entropy": df["cluster_entropy"].mean(),
        "mean_entailment_score": df["entailment_score"].mean()
    }
    
    with open(output_file.replace(".csv", "_summary.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    return metrics

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'semantic_evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
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
        dataset = get_dataset("squad")["validation"]  # Using validation set for evaluation
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