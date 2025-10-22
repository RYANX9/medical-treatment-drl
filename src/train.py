"""
Training Script for Medical Treatment RL Agents
Author: Ahmed Messaad
Usage: python train.py --algo PPO --timesteps 150000
"""

import os
import argparse
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from environment import MedicalTreatmentEnv


# Lab metadata (MIMIC-III)
LAB_ITEM_INFO = {
    50931: {'name': 'Glucose', 'normal_range': (70.0, 100.0), 'unit': 'mg/dL', 'critical': True},
    50912: {'name': 'Creatinine', 'normal_range': (0.6, 1.2), 'unit': 'mg/dL', 'critical': True},
    50902: {'name': 'Chloride', 'normal_range': (98.0, 106.0), 'unit': 'mEq/L', 'critical': False},
    50882: {'name': 'Bicarbonate', 'normal_range': (22.0, 29.0), 'unit': 'mEq/L', 'critical': False},
    50971: {'name': 'Potassium', 'normal_range': (3.5, 5.0), 'unit': 'mEq/L', 'critical': True},
    50983: {'name': 'Sodium', 'normal_range': (136.0, 145.0), 'unit': 'mEq/L', 'critical': True},
    51006: {'name': 'Urea Nitrogen', 'normal_range': (7.0, 20.0), 'unit': 'mg/dL', 'critical': False},
    50868: {'name': 'Anion Gap', 'normal_range': (8.0, 16.0), 'unit': 'mEq/L', 'critical': False},
    51265: {'name': 'Platelet Count', 'normal_range': (150.0, 400.0), 'unit': 'K/uL', 'critical': False},
    51221: {'name': 'Hematocrit', 'normal_range': (36.0, 46.0), 'unit': '%', 'critical': False},
    51301: {'name': 'WBC', 'normal_range': (4.5, 11.0), 'unit': 'K/uL', 'critical': True},
    51222: {'name': 'Hemoglobin', 'normal_range': (13.5, 17.5), 'unit': 'g/dL', 'critical': True}
}


class ProgressCallback(BaseCallback):
    """Track training progress"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        
    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                self.episode_rewards.append(info['r'])
        return True


def load_data(data_path):
    """Load and preprocess MIMIC-III data"""
    print(f"Loading data from {data_path}...")
    
    # This is a placeholder - replace with your actual data loading
    # For the repo, you'd load from preprocessed pickle/CSV files
    raise NotImplementedError(
        "Replace this with your actual data loading logic. "
        "Expected: return train_episodes, val_episodes as List[Dict]"
    )


def train_agent(algo_name, train_episodes, val_episodes, timesteps, save_dir):
    """Train a single RL agent"""
    print(f"\n{'='*80}")
    print(f"Training {algo_name}")
    print(f"{'='*80}")
    
    # Create environments
    train_env = DummyVecEnv([lambda: MedicalTreatmentEnv(train_episodes, LAB_ITEM_INFO)])
    val_env = DummyVecEnv([lambda: MedicalTreatmentEnv(val_episodes, LAB_ITEM_INFO)])
    
    # Hyperparameters
    configs = {
        'PPO': {
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10
        },
        'A2C': {
            'learning_rate': 2e-4,
            'gamma': 0.99,
            'n_steps': 5
        },
        'DQN': {
            'learning_rate': 1e-4,
            'gamma': 0.99,
            'buffer_size': 100000,
            'batch_size': 64
        }
    }
    
    # Initialize model
    algo_class = {'PPO': PPO, 'A2C': A2C, 'DQN': DQN}[algo_name]
    model = algo_class("MlpPolicy", train_env, verbose=1, **configs[algo_name])
    
    # Callbacks
    os.makedirs(save_dir, exist_ok=True)
    eval_callback = EvalCallback(
        val_env,
        eval_freq=5000,
        n_eval_episodes=10,
        best_model_save_path=save_dir,
        deterministic=True
    )
    progress_callback = ProgressCallback()
    
    # Train
    model.learn(total_timesteps=timesteps, callback=[eval_callback, progress_callback])
    
    # Save final model
    model.save(f"{save_dir}/{algo_name}_final.zip")
    print(f"✅ {algo_name} training complete! Saved to {save_dir}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train Medical Treatment RL Agent')
    parser.add_argument('--algo', type=str, default='A2C', choices=['PPO', 'A2C', 'DQN'])
    parser.add_argument('--timesteps', type=int, default=150000)
    parser.add_argument('--data', type=str, default='../data/processed_episodes.pkl')
    parser.add_argument('--save-dir', type=str, default='../models')
    
    args = parser.parse_args()
    
    # Load data
    train_episodes, val_episodes = load_data(args.data)
    
    # Train
    model = train_agent(
        args.algo,
        train_episodes,
        val_episodes,
        args.timesteps,
        args.save_dir
    )
    
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
