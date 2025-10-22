"""
Medical Treatment Environment for Reinforcement Learning
Author: Ahmed Messaad
Description: Custom Gym environment for ICU lab value management using MIMIC-III data
"""

import gym
from gym import spaces
import numpy as np
import random
from typing import List, Dict, Tuple


class MedicalTreatmentEnv(gym.Env):
    """
    Medical Treatment Environment for Sequential Decision Making
    
    State Space (26 dimensions):
    - Lab values (10): Current measurements for each lab test
    - Normalized values (10): Z-scores relative to normal ranges
    - Short-term trends (5): Recent changes in key labs
    - Time (1): Normalized time in episode
    
    Action Space (4 discrete actions):
    - 0: No intervention (watchful waiting)
    - 1: Medication A (e.g., insulin for glucose control)
    - 2: Medication B (e.g., diuretic for fluid balance)
    - 3: Medication C (broad-spectrum stabilization)
    """
    
    def __init__(self, episodes_list: List[Dict], lab_item_info: Dict):
        super().__init__()
        
        self.episodes_list = episodes_list
        self.lab_item_info = lab_item_info
        
        # State space: [10 labs + 10 normalized + 5 trends + 1 time] = 26 features
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(26,),
            dtype=np.float32
        )
        
        # Action space: 4 discrete treatment options
        self.action_space = spaces.Discrete(4)
        
        # Episode state
        self.current_episode = None
        self.current_step = 0
        self.state_history = []
        self.history_window = 5
        
    def reset(self):
        """Reset environment for new episode"""
        self.current_episode = random.choice(self.episodes_list)
        self.current_step = 0
        self.state_history = []
        
        obs = self._get_observation()
        return obs
    
    def _get_observation(self) -> np.ndarray:
        """Construct 26-dimensional observation from current state"""
        current_labs = self.current_episode['lab_values'][self.current_step]
        lab_items = self.current_episode['lab_items']
        
        # Ensure we have 10 values (pad if necessary)
        if len(current_labs) < 10:
            current_labs = np.pad(current_labs, (0, 10 - len(current_labs)), 'constant')
            lab_items = lab_items + [0] * (10 - len(lab_items))
        
        # 1. Raw lab values
        raw_values = current_labs[:10]
        
        # 2. Normalized values (z-scores)
        normalized = np.zeros(10)
        for i in range(min(10, len(lab_items))):
            itemid = lab_items[i]
            if itemid in self.lab_item_info:
                normal_range = self.lab_item_info[itemid]['normal_range']
                mean_normal = (normal_range[0] + normal_range[1]) / 2
                std_normal = (normal_range[1] - normal_range[0]) / 4
                normalized[i] = (raw_values[i] - mean_normal) / (std_normal + 1e-6)
        
        # 3. Trend indicators
        trends = np.zeros(5)
        if len(self.state_history) > 0:
            prev_labs = self.state_history[-1][:10]
            for i in range(5):
                if prev_labs[i] != 0:
                    trends[i] = (raw_values[i] - prev_labs[i]) / (abs(prev_labs[i]) + 1e-6)
        
        # 4. Time feature
        time_norm = self.current_step / max(len(self.current_episode['lab_values']) - 1, 1)
        
        # Combine all features
        obs = np.concatenate([
            raw_values,
            normalized,
            trends,
            [time_norm]
        ]).astype(np.float32)
        
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        obs = np.clip(obs, -10.0, 10.0)
        
        return obs
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action and transition to next state"""
        current_labs = self.current_episode['lab_values'][self.current_step].copy()
        lab_items = self.current_episode['lab_items']
        
        # Store in history
        self.state_history.append(current_labs.copy())
        if len(self.state_history) > self.history_window:
            self.state_history.pop(0)
        
        # Apply treatment effect
        if action > 0:
            current_labs = self._apply_treatment(current_labs, action, lab_items)
        
        # Calculate reward
        reward = self._calculate_reward(current_labs, lab_items, action)
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.current_episode['lab_values']) - 1
        
        # Terminal bonus
        if done:
            final_abnormal = self._count_abnormal(current_labs, lab_items)
            if final_abnormal == 0:
                reward += 20.0
            elif final_abnormal == 1:
                reward += 15.0
            elif final_abnormal == 2:
                reward += 10.0
            elif final_abnormal == 3:
                reward += 5.0
            elif final_abnormal == 4:
                reward += 2.0
        
        # Get next observation
        if not done:
            obs = self._get_observation()
        else:
            obs = np.zeros(26, dtype=np.float32)
        
        info = {
            'abnormal_count': self._count_abnormal(current_labs, lab_items),
            'episode_length': len(self.current_episode['lab_values'])
        }
        
        return obs, reward, done, info
    
    def _apply_treatment(self, labs: np.ndarray, action: int, lab_items: List) -> np.ndarray:
        """Apply treatment effects to lab values"""
        labs = labs.copy()
        
        if action == 1:  # Medication A
            target_items = [50809]
            effect_strength = 0.15
            variability = 0.05
        elif action == 2:  # Medication B
            target_items = [50971, 50983]
            effect_strength = 0.10
            variability = 0.08
        else:  # action == 3
            target_items = lab_items
            effect_strength = 0.05
            variability = 0.10
        
        for i, itemid in enumerate(lab_items):
            if i < len(labs) and itemid in target_items:
                if itemid in self.lab_item_info:
                    normal_range = self.lab_item_info[itemid]['normal_range']
                    target = (normal_range[0] + normal_range[1]) / 2
                    
                    delta = effect_strength * (target - labs[i])
                    noise = np.random.normal(0, variability * abs(labs[i] + 1e-6))
                    labs[i] += delta + noise
                    labs[i] = max(0.1, labs[i])
        
        return labs
    
    def _calculate_reward(self, labs, lab_items, action):
        """Calculate reward based on clinical outcomes"""
        abnormal_count = 0
        normal_count = 0
        improvement_bonus = 0.0
        
        for i, itemid in enumerate(lab_items):
            if i < len(labs) and itemid in self.lab_item_info:
                normal_range = self.lab_item_info[itemid]['normal_range']
                val = labs[i]
                
                if normal_range[0] <= val <= normal_range[1]:
                    normal_count += 1
                else:
                    abnormal_count += 1
                    
                    # Check for improvement
                    if len(self.state_history) > 0 and i < len(self.state_history[-1]):
                        prev_val = self.state_history[-1][i]
                        was_abnormal = (prev_val < normal_range[0] or prev_val > normal_range[1])
                        
                        if was_abnormal:
                            prev_dist = min(abs(prev_val - normal_range[0]), 
                                          abs(prev_val - normal_range[1]))
                            curr_dist = min(abs(val - normal_range[0]), 
                                          abs(val - normal_range[1]))
                            
                            if curr_dist < prev_dist:
                                improvement_bonus += 2.0
        
        # Base reward
        base_reward = (normal_count * 3.0) - (abnormal_count * 1.0)
        
        # Treatment cost
        if action > 0:
            if abnormal_count <= 2:
                treatment_cost = -2.0
            else:
                treatment_cost = -0.3
        else:
            treatment_cost = 0.5 if abnormal_count <= 2 else 0.0
        
        total_reward = base_reward + treatment_cost + improvement_bonus
        return float(total_reward)
    
    def _count_abnormal(self, labs: np.ndarray, lab_items: List) -> int:
        """Count number of abnormal lab values"""
        count = 0
        for i, itemid in enumerate(lab_items):
            if i < len(labs) and itemid in self.lab_item_info:
                normal_range = self.lab_item_info[itemid]['normal_range']
                if labs[i] < normal_range[0] or labs[i] > normal_range[1]:
                    count += 1
        return count
