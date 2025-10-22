# Clinical Safety Analysis: Rule-Based Filtering for RL Treatment Policies

**Author:** Ahmed Messaad  
**Date:** October 2025  
**Context:** Medical Treatment Optimization with Deep Reinforcement Learning (MIMIC-III)

---

## 1. Introduction

Reinforcement learning agents optimize for rewards, not safety. In medical applications, this creates a fundamental problem: an agent might learn to maximize cumulative reward through strategies that violate basic clinical principles. 

This document analyzes the safety characteristics of three trained RL agents (PPO, A2C, DQN) and evaluates the impact of rule-based filtering on clinical appropriateness.

---

## 2. The Over-Treatment Problem

### 2.1 Raw Policy Behavior

When I evaluated the trained agents without any constraints, I observed a concerning pattern:

| Model | Treatment Rate | Appropriateness |
|-------|---------------|-----------------|
| PPO   | 56.8%         | 55.0%           |
| A2C   | 100.0%        | 99.4%           |
| DQN   | 55.8%         | 54.7%           |

**Key Finding:** PPO and DQN treated patients inappropriately 45% of the time. Most errors were treating stable patients who didn't need intervention.

### 2.2 Why This Happens

The reward function penalizes abnormal lab values. Even when a patient is stable with 1-2 minor abnormalities, the agent might treat them "just in case" to avoid future penalties. This is reward hacking—technically optimal under the reward function but clinically inappropriate.

A2C was the exception. It learned an aggressive but appropriate policy: treat nearly every timestep, but only when justified. I'm not entirely sure why A2C succeeded where PPO/DQN failed. My hypothesis is that A2C's on-policy learning made it more sensitive to the treatment cost penalties in the reward function.

---

## 3. Safety Filter Design

### 3.1 Clinical Decision Rules

I implemented three simple rules based on the number of abnormal lab values:
```python
if abnormal_count < 2:
    # Patient is stable → No treatment
    return 0
    
elif abnormal_count >= 3:
    # Patient is sick → Ensure treatment
    if rl_action == 0:
        return 1  # Override RL if it suggests doing nothing
    else:
        return rl_action  # Trust RL's choice of which medication
        
else:
    # Borderline case (2 abnormal) → Trust RL
    return rl_action
```

**Rationale:**
- **< 2 abnormal:** One abnormal lab could be measurement noise or individual variation. Treatment isn't justified.
- **≥ 3 abnormal:** Multiple abnormal values indicate a real problem requiring intervention.
- **= 2 abnormal:** This is a judgment call. Let the RL agent decide.

### 3.2 Limitations of This Approach

This filter has several weaknesses:

1. **Ignores severity:** A glucose of 300 mg/dL (critically high) gets the same weight as a glucose of 105 mg/dL (mildly elevated).

2. **No lab prioritization:** The filter treats WBC and chloride equally, even though WBC abnormalities are more clinically significant.

3. **No temporal context:** It doesn't consider whether a value is improving or worsening.

4. **Arbitrary thresholds:** Why is 3 abnormal the cutoff? This was chosen empirically by testing different values and checking appropriateness scores.

A better approach would learn these thresholds from expert demonstrations or incorporate domain knowledge about lab value severity.

---

## 4. Results

### 4.1 Impact on Clinical Appropriateness

| Model | Raw Appropriateness | Filtered Appropriateness | Improvement |
|-------|---------------------|-------------------------|-------------|
| PPO   | 55.0%               | 94.5%                   | +39.5%      |
| A2C   | 99.4%               | 98.8%                   | -0.6%       |
| DQN   | 54.7%               | 95.5%                   | +40.8%      |

**Interpretation:**
- PPO and DQN needed the filter. Without it, they overtreated stable patients constantly.
- A2C barely changed. It already learned an appropriate policy during training.
- The slight decrease for A2C (-0.6%) is because the filter occasionally overrode correct decisions in borderline cases.

### 4.2 Impact on Reward

| Model | Raw Reward | Filtered Reward | Change  |
|-------|-----------|-----------------|---------|
| PPO   | 152.27    | 148.94          | -3.33   |
| A2C   | 169.01    | 177.02          | +8.01   |
| DQN   | 176.11    | 173.51          | -2.60   |

The reward changes were small. This is expected—the filter only modifies actions when the policy is already making mistakes. For PPO/DQN, preventing inappropriate treatment slightly reduced reward because the reward function isn't perfectly aligned with clinical appropriateness. For A2C, the filter helped by preventing rare edge case errors.

### 4.3 Treatment Rate Changes

| Model | Raw Treatment Rate | Filtered Treatment Rate |
|-------|-------------------|------------------------|
| PPO   | 56.8%             | 98.4%                  |
| A2C   | 100.0%            | 100.0%                 |
| DQN   | 55.8%             | 98.0%                  |

PPO and DQN's treatment rates nearly doubled. This isn't because the filter made them more aggressive—it's because they were previously undertreating sick patients and overtreating healthy ones. The filter corrected both errors, shifting treatment toward patients who actually needed it.

---

## 5. Case Study: PPO Before and After Filtering

I tracked PPO's decisions across one test episode to understand the filter's impact:

### Episode 42: 18 Timesteps

**Patient Profile:**
- Initial state: 6 abnormal labs (glucose, WBC, creatinine elevated)
- Duration: 3.2 hours

**Raw PPO Policy:**
- Steps 1-8: Treated aggressively (Action 1, 2, 3)
- Steps 9-12: Stopped treating when 3 abnormal remained
- Steps 13-18: Resumed treatment sporadically

**Problem:** At step 10, the patient still had 3 abnormal values but PPO chose Action 0 (do nothing). This is clinically inappropriate.

**Filtered PPO Policy:**
- Steps 1-8: Same as raw policy
- Steps 9-12: Filter overrode Action 0 → Action 1 (glucose control)
- Steps 13-18: Continued treatment until abnormal count < 2

**Outcome:**
- Raw: Final abnormal count = 4
- Filtered: Final abnormal count = 2

The filter prevented premature treatment cessation.

---

## 6. Discussion

### 6.1 Why A2C Succeeded Without Filtering

I don't have a definitive answer, but I have two hypotheses:

1. **On-policy learning:** A2C only learns from its current policy. This might make it more sensitive to the treatment cost penalties in the reward function. PPO and DQN, with their experience replay mechanisms, might average over past overtreating behaviors.

2. **Exploration strategy:** A2C's entropy coefficient was 0.01 (same as PPO), but its training dynamics might have led to different exploration patterns. It's possible A2C stumbled into the "treat aggressively but appropriately" strategy early in training and reinforced it.

This needs more investigation. I'd want to do ablation studies varying the entropy coefficient, reward weights, and training length to see what factors matter most.

### 6.2 Limitations of This Work

Several important caveats:

1. **Simulated treatment effects:** The environment applies treatment effects using fixed formulas. Real patients don't respond this predictably.

2. **Small dataset:** 120 episodes isn't enough to generalize. The agents might be overfitting to quirks of these specific patients.

3. **No validation with clinicians:** I defined "appropriate" based on abnormal counts, but real clinicians consider severity, trends, and patient history.

4. **Reward function issues:** The fact that raw PPO/DQN policies are inappropriate suggests my reward function is misspecified. A better reward would make the filter unnecessary.

5. **No comparison to expert policies:** I don't know how human physicians would perform on these same episodes.

### 6.3 Future Directions

If I continued this project, I'd focus on:

1. **Inverse reinforcement learning:** Learn the reward function from expert demonstrations instead of hand-designing it.

2. **Constrained policy optimization:** Use algorithms like CPO (Constrained Policy Optimization) that enforce safety constraints during training rather than post-hoc filtering.

3. **Hierarchical policies:** Separate high-level decisions ("should we treat?") from low-level decisions ("which medication?").

4. **Uncertainty quantification:** Make the agent admit when it's unsure instead of guessing.

---

## 7. Conclusion

Rule-based safety filtering improved clinical appropriateness by 40 percentage points for PPO and DQN. However, this is a band-aid solution. The real problem is that the reward function doesn't fully capture what we want the agent to do.

A2C learned an appropriate policy without filtering, which suggests that better reward design or algorithm choice could eliminate the need for post-hoc constraints. But until we understand why A2C succeeded, safety filters remain a necessary safeguard for deploying RL agents in medical decision-making.

The key takeaway: **RL agents optimize for the reward you give them, not the behavior you want.** In safety-critical domains, assume your reward function is wrong and build in guardrails.

---

**Acknowledgments:** This work uses the MIMIC-III Clinical Database (Johnson et al., 2016). Thanks to the MIT Lab for Computational Physiology for making this data publicly available.
