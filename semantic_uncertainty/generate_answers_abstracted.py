"""Generate answers with LLM, cache activations, and compute entropy with few-shot prompting."""
import numpy as np
import torch
from datasets import load_dataset
from uncertainty.utils import utils
import hashlib
import random

def main(args):
    # Load and split dataset
    dataset = load_dataset("openbmb/UltraFeedback")["train"].train_test_split(test_size=0.2, seed=42)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # Reformat dataset
    def reformat(example, j):
        try:
            md5hash = lambda s: str(int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16))
            return {
                'question': 'Evaluate the following model response: ' + example['completions'][j].get('response', 'No response found'),
                'context': example['instruction'],
                'id': md5hash(str(example['instruction']))
            }
        except:
            return None

    train_dataset = [x for d in train_dataset for j in range(4) if (x := reformat(d, j)) is not None]
    test_dataset = [x for d in test_dataset for j in range(4) if (x := reformat(d, j)) is not None]

    # Construct few-shot prompt from training data
    def construct_fewshot_prompt(dataset, num_examples=5):
        prompt = """You are an evaluator of text quality. Below are examples of instructions and responses, followed by an evaluation. Use these to guide your answers.\n\n"""
        sampled_indices = random.sample(range(len(dataset)), min(num_examples, len(dataset)))
        for idx in sampled_indices:
            example = dataset[idx]
            question = example['question']
            context = example['context']
            # Use the original response as a pseudo-answer for demonstration (since we don’t have ground truth evaluations here)
            pseudo_answer = "Rating: 4\nRationale: The response aligns well with the instruction, with minor deviations."
            prompt += f"Instruction: {context}\nQuestion: {question}\nAnswer: {pseudo_answer}\n\n"
        prompt += "Now, provide your evaluation for the following:\n"
        return prompt

    few_shot_prompt = construct_fewshot_prompt(train_dataset, num_examples=args.num_few_shot)
    print("Few-shot prompt constructed:")
    print(few_shot_prompt)

    # Initialize model
    model = utils.init_model(args)

    # Process each split
    for dataset_split, dataset in [('train', train_dataset), ('validation', test_dataset)]:
        print(f"Generating answers for {dataset_split} split")
        generations = {}

        # Limit to a small subset for efficiency (adjustable via args.num_samples)
        indices = range(min(args.num_samples, len(dataset)))

        for index in indices:
            example = dataset[index]
            question, context = example["question"], example['context']
            generations[example['id']] = {'question': question, 'context': context}

            # Combine few-shot prompt with current input
            current_input = f"Instruction: {context}\nQuestion: {question}\nAnswer:"
            local_prompt = few_shot_prompt + current_input

            # Generate 5 answers
            full_responses = []
            num_generations = 5  # Start with 5, can be set to 10
            print(f"\nGenerating answers for ID: {example['id']}")
            print(f"Question: {question}")
            for i in range(num_generations):
                temperature = args.temperature  # Consistent temperature (e.g., 1.0)
                predicted_answer, token_log_likelihoods, (embedding, _, _) = model.predict(
                    local_prompt, temperature, return_latent=True
                )
                embedding = embedding.cpu() if embedding is not None else None
                full_responses.append((predicted_answer, token_log_likelihoods, embedding))
                # Print the generated answer
                print(f"Answer {i + 1}: {predicted_answer}")

            # Compute entropy based on unique responses
            responses = [r[0] for r in full_responses]
            unique_responses, counts = np.unique(responses, return_counts=True)
            probs = counts / len(responses)
            entropy = -np.sum(probs * np.log(probs)) if len(probs) > 0 else 0
            print(f"Entropy: {entropy:.4f}")

            # Store results
            generations[example['id']]['responses'] = full_responses
            generations[example['id']]['entropy'] = entropy

        # Save generations
        utils.save(generations, f'{dataset_split}_generations.pkl', save_dir="/workspace/saved")

    print("Run complete.")
    del model

if __name__ == '__main__':
    parser = utils.get_parser()
    # Add num_few_shot to parser since it’s now used
    parser.add_argument("--num_few_shot", type=int, default=5, help="Number of few-shot examples")
    args = parser.parse_args()
    print(f"Starting run with args: {args}")
    main(args)
