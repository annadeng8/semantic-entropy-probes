"""Generate evaluations with LLM, cache activations, and compute entropy with few-shot prompting."""
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

    # Reformat dataset to include responses and annotations
    def reformat(example, j):
        try:
            completion = example['completions'][j]
            response = completion.get('response', 'No response found')
            annotations = completion.get('annotations', {})
            instruction_following = annotations.get('instruction_following', {})
            rating = instruction_following.get('Rating', 'Unknown rating')
            rationale = instruction_following.get('Rationale', 'No rationale provided')
            md5hash = lambda s: str(int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16))
            return {
                'question': example['instruction'],
                'response': response,
                'evaluation': f"Rating: {rating}\nRationale: {rationale}",
                'id': md5hash(str(example['instruction']))
            }
        except:
            return None

    train_dataset = [x for d in train_dataset for j in range(4) if (x := reformat(d, j)) is not None]
    test_dataset = [x for d in test_dataset for j in range(4) if (x := reformat(d, j)) is not None]

    # Construct few-shot prompt using Instruction, Question, Answer, and Rationale
    def construct_fewshot_prompt(dataset, num_examples=10):
        prompt = """You are an evaluator of text quality. Below are examples to guide your evaluation. Each example includes an Instruction (the task), Question (the specific query), Answer (the response), and Rationale (the reasoning behind the rating). Use these to provide your evaluation.\n\n"""
        sampled_indices = random.sample(range(len(dataset)), min(num_examples, len(dataset)))
        for idx in sampled_indices:
            example = dataset[idx]
            instruction = example['question']  # Instruction is the same as the question
            question = example['question']     # Question is the same as instruction
            answer = example['response']
            evaluation = example['evaluation']
            prompt += f"Instruction: {instruction}\nQuestion: {question}\nAnswer: {answer}\nEvaluation: {evaluation}\n\n"
        prompt += "Now, provide your evaluation for the following AND GIVE NOTHING ELSE:\n"
        return prompt

    # Initialize model
    model = utils.init_model(args)

    # Process each split
    for dataset_split, dataset in [('train', train_dataset), ('validation', test_dataset)]:
        print(f"Generating evaluations for {dataset_split} split")
        generations = {}

        # Limit to a small subset for efficiency
        indices = range(min(args.num_samples, len(dataset)))

        for index in indices:
            example = dataset[index]
            question = example["question"]
            test_answer = example["response"]  # Use the dataset's response as the answer to evaluate
            # Minimal change: use "context" for the original instruction, and prepend the evaluation prompt to the response
            generations[example['id']] = {
                'context': question,
                'question': "Evaluate the following model response: " + test_answer,
                'responses': []  # initialize the responses key
            }

            # Construct few-shot prompt for this specific example
            few_shot_prompt = construct_fewshot_prompt(train_dataset, num_examples=args.num_few_shot)

            # Combine few-shot prompt with current input
            current_input = f"Instruction: {question}\nQuestion: {question}\nAnswer: {test_answer}\nEvaluation:"
            local_prompt = few_shot_prompt + current_input

            # Print the full prompt before generating evaluations
            print("Few-shot prompt constructed:")
            print(local_prompt)

            # Generate 5 evaluations
            full_evaluations = []
            ratings = []
            num_generations = 10
            for i in range(num_generations):
                temperature = args.temperature
                predicted_evaluation, token_log_likelihoods, (embedding, _, _) = model.predict(
                    local_prompt, temperature, return_latent=True
                )
                embedding = embedding.cpu() if embedding is not None else None
                # Minimal change: append a tuple containing the evaluation string, token log likelihoods, and embedding
                full_evaluations.append((predicted_evaluation, token_log_likelihoods, embedding))
                # Extract rating from predicted_evaluation
                try:
                    rating_str = predicted_evaluation.split("Rating: ")[1].split("\n")[0].strip()
                    rating = int(rating_str)
                except (IndexError, ValueError):
                    rating = None  # Handle cases where rating extraction fails
                ratings.append(rating)
                evaluation_one_line = predicted_evaluation.replace('\n', ' ')
                print(f"Evaluation {i + 1}: {evaluation_one_line}")

            # Compute entropy based on ratings
            valid_ratings = [r for r in ratings if r is not None]
            if valid_ratings:
                unique_ratings, counts = np.unique(valid_ratings, return_counts=True)
                probs = counts / len(valid_ratings)
                entropy = -np.sum(probs * np.log(probs))
            else:
                entropy = 0  # Default entropy if no valid ratings
            print(f"Entropy: {entropy:.4f}")

            # Minimal change: use key "responses" instead of "evaluations"
            generations[example['id']]['responses'] = full_evaluations
            generations[example['id']]['entropy'] = entropy

        # Save generations
        utils.save(generations, f'{dataset_split}_generations.pkl', save_dir="/workspace/saved")

    print("Run complete.")
    del model

if __name__ == '__main__':
    parser = utils.get_parser()
    parser.add_argument("--num_few_shot", type=int, default=5, help="Number of few-shot examples")
    args = parser.parse_args()
    print(f"Starting run with args: {args}")
    main(args)
