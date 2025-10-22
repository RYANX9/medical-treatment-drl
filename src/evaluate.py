"""
Evaluation Script for Medical Treatment RL Agents
Author: Ahmed Messaad
Usage: python evaluate.py --model ../models/A2C_final.zip --episodes 50
"""

import argparse
import numpy as np
import json
from stable_baselines3 import PPO, A2C, DQN
from environment import MedicalTreatmentEnv
from train import LAB_ITEM_INFO


class ClinicalSafetyFilter:
    """Wrapper that enforces clinical safety constraints"""
    def __init__(self, base_model, treat_threshold=3, stable_threshold=2):
        self.base_model = base_model
        self.treat_threshold = treat_threshold
        self.stable_threshold = stable_threshold
        
    def predict(self, obs, abnormal_count, deterministic=True):
        """Get clinically-filtered action"""
        rl_action, _ = self.base_model.predict(obs, deterministic=deterministic)
        
        if abnormal_count < self.stable_threshold:
            return 0, None  # Don't treat stable patients
        elif abnormal_count >= self.treat_threshold:
            if rl_action == 0:
                return 1, None  # Ensure treatment for sick patients
            else:
                return rl_action, None
        else:
            return rl_action, None  # Trust RL for borderline cases


def evaluate_model(model, env, n_episodes, use_filter=False):
    """Comprehensive model evaluation"""
    all_rewards = []
    all_episode_rewards = []
    all_actions = []
    all_abnormal_counts = []
    
    if use_filter:
        policy = ClinicalSafetyFilter(model)
    else:
        policy = model
    
    for ep_idx in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_actions = []
        episode_abnormal = []
        
        info = {'abnormal_count': 3}  # Initialize
        
        while not done:
            if use_filter:
                abnormal_count = info.get('abnormal_count', 3)
                action, _ = policy.predict(obs, abnormal_count)
            else:
                action, _ = policy.predict(obs, deterministic=True)
            
            obs, reward, done, info = env.step(action)
            
            all_rewards.append(reward)
            episode_reward += reward
            episode_actions.append(int(action))
            episode_abnormal.append(info.get('abnormal_count', 0))
        
        all_episode_rewards.append(episode_reward)
        all_actions.extend(episode_actions)
        all_abnormal_counts.extend(episode_abnormal)
    
    # Calculate appropriateness
    appropriate = sum(
        1 for action, abnormal in zip(all_actions, all_abnormal_counts)
        if (action > 0 and abnormal >= 3) or (action == 0 and abnormal < 2)
    )
    appropriateness = appropriate / len(all_actions) if len(all_actions) > 0 else 0
    
    return {
        'mean_reward': np.mean(all_episode_rewards),
        'std_reward': np.std(all_episode_rewards),
        'mean_abnormal': np.mean(all_abnormal_counts),
        'treatment_rate': np.mean(np.array(all_actions) > 0),
        'appropriateness': appropriateness,
        'action_distribution': np.bincount(all_actions, minlength=4).tolist()
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate Medical Treatment RL Agent')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--algo', type=str, default='A2C', choices=['PPO', 'A2C', 'DQN'])
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--data', type=str, default='../data/processed_episodes.pkl')
    parser.add_argument('--filter', action='store_true', help='Use clinical safety filter')
    
    args = parser.parse_args()
    
    # Load model
    algo_class = {'PPO': PPO, 'A2C': A2C, 'DQN': DQN}[args.algo]
    model = algo_class.load(args.model)
    
    # Load test data (placeholder - replace with actual loading)
    # test_episodes = load_test_data(args.data)
    test_episodes = []  # Replace this
    
    # Create environment
    env = MedicalTreatmentEnv(test_episodes, LAB_ITEM_INFO)
    
    # Evaluate
    print(f"\n{'='*80}")
    print(f"Evaluating {args.algo} {'with' if args.filter else 'without'} clinical filter")
    print(f"{'='*80}\n")
    
    results = evaluate_model(model, env, args.episodes, use_filter=args.filter)
    
    # Display results
    print(f"Mean Episode Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Mean Abnormal Labs: {results['mean_abnormal']:.2f}")
    print(f"Treatment Rate: {results['treatment_rate']:.1%}")
    print(f"Clinical Appropriateness: {results['appropriateness']:.1%}")
    print(f"Action Distribution: {results['action_distribution']}")
    
    # Save results
    output_file = f"../results/evaluation_{'filtered' if args.filter else 'raw'}_{args.algo}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n✅ Results saved to {output_file}")


if __name__ == "__main__":
    main()
