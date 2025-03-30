"""Predict with LLM on task."""
import gc
import os
import logging
import random
from tqdm import tqdm
import hashlib

import numpy as np
import torch

from datasets import load_dataset
from uncertainty.utils import utils
from uncertainty.uncertainty_measures import p_true as p_true_utils
from compute_uncertainty_measures import main as main_compute


utils.setup_logger()


def main(args):
    experiment_details = {'args': args}
    random.seed(args.random_seed)

    metric = utils.get_metric('llm')

    
    # Load dataset
    dataset = load_dataset("openbmb/UltraFeedback")
    # Print available dataset splits
    print(dataset)

    # Use the train-test split
    # Split the dataset into train (80%) and test (20%) with a fixed seed for reproducibility
    dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)

    # Extract train and test sets
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    
    def reformat(example, j):
        try:
            md5hash = lambda s: str(int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16))
            return {
                'question': 'Evaluate the following model response: '+example['completions'][j].get('response', 'No response found'), 
                'context': example['instruction'],
                'answers': {'text': [str(example['completions'][j].get('annotations'))]},
                'id': md5hash(str(example['instruction']))
            }
        except:
            print("ERROR")
    

    train_dataset = [x for d in train_dataset for j in range(4) if (x := reformat(d, j)) is not None]
    test_dataset = [x for d in test_dataset for j in range(4) if (x := reformat(d, j)) is not None]

    # Get indices of answerable and unanswerable questions and construct prompt.
    answerable_indices, unanswerable_indices = utils.split_dataset(train_dataset)

    unanswerable_indices = []
    val_answerable, val_unanswerable = utils.split_dataset(test_dataset)
    del val_unanswerable
    test_dataset = [test_dataset[i] for i in val_answerable]

    prompt_indices = random.sample(answerable_indices, args.num_few_shot)
    experiment_details['prompt_indices'] = prompt_indices
    remaining_answerable = list(set(answerable_indices) - set(prompt_indices))

    print("creating few-shot prompt")
    # Create Few-Shot prompt.
    make_prompt = utils.get_make_prompt(args)
    prompt = utils.construct_fewshot_prompt_from_indices(
        train_dataset, prompt_indices, make_prompt)
    experiment_details['prompt'] = prompt
    logging.info('Prompt is: %s', prompt)
    print('Prompt is: %s', prompt)

    # Initialize model.
    print("initialize model")
    model = utils.init_model(args)

    if args.compute_p_true:
        logging.info(80*'#')
        logging.info('Constructing few-shot prompt for p_true.')
        print(80*'#')
        print('Constructing few-shot prompt for p_true.')

        p_true_indices = random.sample(answerable_indices, args.p_true_num_fewshot)
        remaining_answerable = list(set(remaining_answerable) - set(p_true_indices))
        p_true_few_shot_prompt, p_true_responses, len_p_true = p_true_utils.construct_few_shot_prompt(
            model=model, dataset=train_dataset, indices=p_true_indices,
            prompt=prompt, brief=None,
            brief_always=args.brief_always and args.enable_brief,
            make_prompt=make_prompt, num_generations=args.num_generations,
            metric=metric)
       
        experiment_details['p_true_indices'] = p_true_indices
        experiment_details['p_true_responses'] = p_true_responses
        experiment_details['p_true_few_shot_prompt'] = p_true_few_shot_prompt
        logging.info('Finished constructing few-shot prompt for p_true.')
        logging.info(80*'#')
        logging.info('p_true_few_shot_prompt: %s', p_true_few_shot_prompt)
        logging.info(80*'#')

        print('Finished constructing few-shot prompt for p_true.')
        print(80*'#')
        print('p_true_few_shot_prompt: %s', p_true_few_shot_prompt)
        print(80*'#')

    # Start answer generation.
    logging.info(80 * '=')
    logging.info('Generating answers: ')
    logging.info(80 * '=')
    print(80 * '=')
    print('Generating answers: ')
    print(80 * '=')
    for dataset_split in ['train', 'validation']:
        logging.info(80 * 'x')
        print(80 * 'x')
        logging.info('Starting with dataset_split %s.', dataset_split)
        print('starting with dataset_split', dataset_split)
        logging.info(80 * 'x')
        print(80 * 'x')

        # This will store all input data and model predictions.
        accuracies, generations, results_dict, p_trues = [], {}, {}, []

        if dataset_split == 'train':
            if not args.get_training_set_generations:
                logging.info('Skip training data.')
                print("skip training data")
                continue
            dataset = train_dataset
            possible_indices = list(set(remaining_answerable) | set(unanswerable_indices))

        else:
            dataset = test_dataset
            possible_indices = range(0, len(dataset))

        # Evaluate over random subset of the datasets.
        indices = random.sample(possible_indices, min(args.num_samples, len(dataset)))
        experiment_details[dataset_split] = {'indices': indices}

        if args.num_samples > len(dataset):
            logging.warning('Not enough samples in dataset. Using all %d samples.', len(dataset))
            print('Not enough samples in dataset. Using all %d samples.', len(dataset))

        it = 0
        for index in tqdm(indices):
            if (it + 1 % 10) == 0:
                gc.collect()
                torch.cuda.empty_cache()
            it += 1

            # Grab example at index.
            example = dataset[index]
            question, context = example["question"], example['context']
            generations[example['id']] = {'question': question, 'context': context}
            correct_answer = example['answers']['text']

            current_input = make_prompt(
                context, question, None)
            local_prompt = prompt + current_input
            print("current input: ".ljust(15) + current_input)
            logging.info('Current input: '.ljust(15) + current_input)

            full_responses = []

            # We sample 1 low temperature answer on which we will compute the
            # accuracy and args.num_generation high temperature answers which will
            # be used to estimate the entropy.

            if dataset_split == 'train' and args.get_training_set_generations_most_likely_only:
                num_generations = 1
            else:
                num_generations = args.num_generations + 1

            for i in range(num_generations):

                # Temperature for first generation is always `0.1`.
                temperature = 0.1 if i == 0 else args.temperature
                print("this is the local prompt", local_prompt)
                predicted_answer, token_log_likelihoods, (embedding, emb_last_before_gen, emb_before_eos) = model.predict(local_prompt, temperature, return_latent=True) 
                
                # Last token embedding
                embedding = embedding.cpu() if embedding is not None else None
                emb_last_before_gen = emb_last_before_gen.cpu() if emb_last_before_gen is not None else None
                emb_before_eos = emb_before_eos.cpu() if emb_before_eos is not None else None
                
                compute_acc = args.compute_accuracy_at_all_temps or (i == 0)
                if correct_answer and compute_acc:
                    acc = metric(predicted_answer, example, model)
                else:
                    acc = 0.0  # pylint: disable=invalid-name

                if i == 0:
                    accuracies.append(acc)
                    most_likely_answer_dict = {
                        'response': predicted_answer,
                        'token_log_likelihoods': token_log_likelihoods,
                        'embedding': embedding,
                        'accuracy': acc,
                        'emb_last_tok_before_gen': emb_last_before_gen,
                        'emb_tok_before_eos': emb_before_eos, 
                    }

                    generations[example['id']].update({
                        'most_likely_answer': most_likely_answer_dict,
                        'reference': utils.get_reference(example),
                    })
                else:
                    # Aggregate predictions over num_generations.
                    full_responses.append(
                        (predicted_answer, token_log_likelihoods, embedding, acc))

            # Append all predictions for this example to `generations`.
            generations[example['id']]['responses'] = full_responses

            if args.compute_p_true and dataset_split == 'validation':
                # Already compute p_true here. Avoid cost of generations in compute_uncertainty script.
                p_true = p_true_utils.calculate_p_true(
                    model, question, most_likely_answer_dict['response'],
                    [r[0] for r in full_responses], p_true_few_shot_prompt,
                    hint=args.p_true_hint)
                p_trues.append(p_true)
                logging.info('p_true: %s', p_true)
                print('p_true:', p_true)

        # Save generations for that split.
        # utils.save(generations, f'{dataset_split}_generations.pkl')
        utils.save(generations, f'{dataset_split}_generations.pkl', save_dir="/workspace/saved")

        # Log overall accuracy.
        accuracy = np.mean(accuracies)
        print(f"Overall {dataset_split} split accuracy: {accuracy}")

        if dataset_split == 'validation':
            if args.compute_p_true:
                results_dict['uncertainty_measures'] = {
                    'p_false':  [1 - p for p in p_trues],
                    'p_false_fixed':  [1 - np.exp(p) for p in p_trues],
                }
            #utils.save(results_dict, 'uncertainty_measures.pkl')
            utils.save(results_dict, 'uncertainty_measures.pkl', save_dir="/workspace/saved")

    utils.save(experiment_details, 'experiment_details.pkl', save_dir="/workspace/saved")
    logging.info('Run complete.')
    del model



if __name__ == '__main__':

    parser = utils.get_parser()
    args, unknown = parser.parse_known_args()
    logging.info('Starting new run with args: %s', args)
    print('Starting new run with args: %s', args)

    if unknown:
        raise ValueError(f'Unkown args: {unknown}')

    logging.info('STARTING `generate_answers`!')
    main(args)
    logging.info('FINISHED `generate_answers`!')