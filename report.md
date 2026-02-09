# The Lazy Artist: Diagnosing and Curing Shortcut Learning on Colored-MNIST

PreCog Recruitment Task (Computer Vision)  
Author: **Lohith Kola**  
Date: **09/02/2026**

---

## What I did and did not do (and why) — First Page

### ✅ Completed

- **Task 0 (The Biased Canvas):** Built a synthetic Colored-MNIST dataset with a **95% spurious color correlation** in the Easy split and **inverted / de-correlated** correlation in the Hard split. I iterated on the dataset design after realizing my first version was not strong enough to induce cheating.
- **Task 1 (The Cheater):** Trained CNNs on the biased Easy split, achieved very high Easy accuracy, and showed a large performance drop on the Hard split. Used confusion matrices + counterfactual recoloring tests to prove shortcut learning.
- **Task 3 (The Interrogation):** Implemented **Grad-CAM from scratch** (no `pytorch-gradcam`) and used it to visually verify what the model attends to on biased vs conflicting examples.
- **Task 4 (The Intervention):** Implemented **at least two training strategies** to make the model focus on shape rather than color without converting to grayscale and without changing the dataset. Also tried a gradient-based color penalty approach, which did not improve baseline performance.
- **Task 5 (The Invisible Cloak):** Performed a **targeted adversarial attack** under an invisibility constraint (ε < 0.05) and compared the lazy vs robust model by required noise magnitude / target confidence.

### ❌ Not completed

- **Task 2 (The Prober):** Not completed. I chose to prioritize end-to-end evidence for shortcut learning + interventions + adversarial evaluation under time constraints.
- **Task 6 (The Decomposition):** Not completed. SAE training and interpretability interventions are tuning-heavy and exploratory, and I prioritized Task 4 and Task 5.

---

## Abstract

CNNs are powerful but often lazy: if they find a shortcut feature that predicts labels during training, they may rely on it instead of learning the intended concept. In this project, I deliberately constructed a dataset that rewards cheating by correlating digit labels with color 95% of the time. A model trained on this “Easy” biased data achieved strong in-distribution performance but collapsed on a “Hard” test split where the correlation is inverted or broken. I diagnosed this shortcut behavior with confusion matrices, counterfactual recoloring tests, and a from-scratch Grad-CAM implementation.

Then I tried to “cure” the model using custom training strategies that discourage reliance on color while keeping the biased dataset unchanged and keeping the images colored. Finally, I evaluated robustness via a targeted adversarial attack under an imperceptibility constraint (ε < 0.05) and compared how easily the lazy vs robust models can be fooled.

---

## 1. My mindset: being a “psychologist” for a neural network

I treated the model like a student who wants to get the highest score with the least effort. If color makes classification easy, it will learn color. So my workflow was:

1. **Create temptation:** build a biased dataset where color is predictive.
2. **Catch cheating quantitatively:** Easy accuracy is high, Hard accuracy collapses.
3. **Catch cheating qualitatively:** interpretability maps + recoloring counterfactuals.
4. **Try interventions:** retrain using strategies that force reliance on shape.
5. **Stress test:** adversarially attack and compare “lazy” vs “robust” models.

---

## 2. Experimental setup (high-level)

- **Base data:** MNIST-derived digit images
- **Easy Train/Val:** 95% digit-to-color correlation (spurious shortcut available)
- **Hard Test:** inverted or randomized mapping (shortcut breaks)
- **Models:**
  - A “stronger” CNN that was initially learning shapes too well
  - A deliberately **lazier CNN** (reduced capacity) to encourage shortcut learning
  - Robust variants trained with Task 4 interventions
- **Metrics and artifacts:**
  - Easy train accuracy, Easy val accuracy, Hard test accuracy
  - Confusion matrix on Hard test
  - Counterfactual recoloring tests (e.g., **Red 1 → predicted 0?**)
  - Grad-CAM overlays
  - Targeted attack success at ε < 0.05, and required noise magnitude for 90% confidence

---

## 3. Task 0 — The Biased Canvas (Dataset)

### Goal

Create a Colored-MNIST dataset that “lies”:

- The **digit** is the true signal.
- The **color** is a spurious shortcut correlated with the digit on Easy split.
- The shortcut breaks on Hard test split.

### Important: my first dataset design failed (and why)

Initially, in Task 0, I built a dataset where the **digit stroke was colored but the background was plain black**. This seemed fine, but it produced a key issue in Task 1:

- The **spurious signal was not stronger than the true digit shape**.
- As a result, a simple 3-layer CNN remained surprisingly accurate even on the Hard test split because it could still learn digit shape.

So the dataset was biased “on paper”, but not biased enough in practice. The model did not fully “traumatize” the way this task intended.

### Fix: make the shortcut louder (texture background)

To force shortcut learning properly, I switched the design to include a **textured background** (as explained in my notebook). This made color-based cues much more dominant and tempting, and it finally created the intended collapse on the Hard split.

### Final dataset rules

- **Easy Train/Val (biased):**  
  For each digit `y ∈ {0..9}`, assign a dominant color `c(y)` with probability **0.95**, otherwise random (5% counterexamples).
- **Hard Test (anti-bias):**  
  Invert or break the correlation:
  - “0 is never Red”
  - “1 is never Green”
  - etc.
- **Twist enforced:**  
  Color is applied as stroke/background texture, not a flat solid fill.

### Sanity checks I used

- Verified correlation statistics per class match expected rates.
- Visual grids per digit per split to confirm the bias is obvious.

---

## 4. Task 1 — The Cheater

### Goal

Train a standard CNN on the Easy set:

- High accuracy on Easy Train and Easy Val
- Accuracy collapses on Hard Test

### Unexpected issue: my first CNN was still learning shape

Initially, I trained a CNN that had enough parameters and depth to learn digit shapes quite well even under heavy color bias. That reduced the intended “cheater effect”.

So I deliberately made the network **lazier**:

- I reduced capacity (e.g., reduced convolution widths / channels).
- This pushed the model into relying on the spurious shortcut because it was the simplest winning strategy.

### Key results

- **Easy Train Accuracy:** 96.1%
- **Easy Val Accuracy:** 95.5%
- **Hard Test Accuracy (Lazy model):** 22.4%

That's a 73-point gap. 96% when the color shortcut is present, 22% when it's removed.

### Diagnostics I used beyond just accuracy

1. **Confusion Matrix on Hard Test**
   - Shows which digits collapse into which predictions under broken color mapping. Each digit gets predicted as the class whose dominant color matches the background.
2. **Counterfactual recoloring test**
   - I took a single grayscale "1" and placed it on every background color. It cheated **8 out of 9 times**: Red → predicted 0, Blue → 2, Yellow → 3, Teal → 5, Pink → 6, Brown → 7, White → 8, Black → 9. The only time it said "1" was on green — because green = 1 in the training data.
   - The key case: a **Red 1** was predicted as **0** at 46% confidence, with class 1 trailing at 40%. It literally _sees_ a "1" and says "0" because red = 0 in the color-label mapping.

### My takeaway

Shortcut learning is not just theoretical. When the spurious cue is strong enough, the model behaves like it is reading a “color answer key”.

---

## 5. Task 3 — The Interrogation (Grad-CAM from scratch)

### Goal

Prove where the model is “looking” without using `pytorch-gradcam`.

### What I implemented

I implemented Grad-CAM by:

1. Hooking into the **final convolution layer** to capture feature maps.
2. Computing gradients of the target class score w.r.t. those feature maps.
3. Global-average pooling gradients to get channel weights.
4. Weighted sum of feature maps and ReLU to create heatmap.
5. Upsampling and overlaying on the original image.

### Implementation references I used

- YouTube (Grad-CAM explained):  
  https://www.youtube.com/watch?v=_QiebC9WxOc
- YouTube (Grad-CAM implementation in PyTorch):  
  https://www.youtube.com/watch?v=eLQZrNYqjNg

### Findings

| Experiment               | What happened                                                                                                                                    |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| Red "0" (biased)         | Predicted 0 at 100% confidence. Heatmap smears across the background — barely touches the digit shape. Right answer, completely wrong reasoning. |
| Red "1" (conflict)       | Predicted **0** at 45% confidence. Same broad background attention. It literally _sees_ a "1" and says "0" because the background is red.        |
| One "1" on all 10 colors | Cheated **8 out of 9** times. Each background color → the model predicts that color's associated class, not "1".                                 |

**Quantitative attention breakdown (digit vs background):**

| Setup                | Digit%   | Background% | Prediction |
| -------------------- | -------- | ----------- | ---------- |
| 0 on red (biased)    | 24.0%    | 76.0%       | 0          |
| 0 on teal (conflict) | 24.9%    | 75.1%       | 5          |
| 1 on green (biased)  | 8.2%     | 91.8%       | 1          |
| 1 on pink (conflict) | **1.0%** | **99.0%**   | 6          |
| 7 on brown (biased)  | 8.0%     | 92.0%       | 7          |
| 7 on blue (conflict) | 18.5%    | 81.5%       | 2          |

The most damning number: on digit "1" placed on a pink background, the model puts **1% of attention on the digit and 99% on the background**. It has essentially learned to be a color classifier that happens to output digit labels.

### Artifacts

- Grad-CAM overlays for biased, conflicting, and hard test examples
- Attention percentage tables (digit vs background pixels)
- Consistency test: same digit "0" on 4 different background colors produces completely different heatmaps and predictions

---

## 6. Task 4 — The Intervention

### Goal

Retrain without grayscale conversion and without changing the biased dataset, but still:

- Achieve **>70% accuracy on Hard test**

### The main idea

If the model is addicted to color, I need to either:

- reduce how useful color is during optimization, or
- add constraints/regularization that penalize color reliance, or
- force the model to learn features that are stable across environments.

### Methods I tried

#### Method A: Gradient-based color penalty (did not work)

I tried a gradient-based penalty designed to reduce sensitivity to color. It did not improve baseline Hard test accuracy.

**My hypothesis why it failed:**

- The penalty might have been too weak relative to classification loss.
- The model might have found alternative color-sensitive features that bypass the penalty.
- The penalty might have targeted the wrong notion of “color dependence”.

#### Method B: Color-Jitter Consistency Loss (the winner)

For each training image **x**, create a color-jittered copy **x̃** (random hue, saturation, brightness shifts — no spatial transforms). Forward both through the model and penalize any difference in predictions via KL divergence:

```
L = L_CE(f(x), y) + α · D_KL( f(x) ‖ f(x̃) )
```

The hyperparameter **α** controls how much we care about color invariance. I tested four values:

| α       | Test Hard | Gap      | Improvement over baseline |
| ------- | --------- | -------- | ------------------------- |
| 0.5     | 55.8%     | 0.40     | +33.2%                    |
| 1.0     | 55.9%     | 0.40     | +33.3%                    |
| **2.0** | **62.5%** | **0.33** | **+39.8%**                |
| 5.0     | 55.5%     | 0.39     | +32.9%                    |

The sweet spot is **α=2.0** — pushing harder (α=5.0) hurts train accuracy without helping generalization. The consistency loss steadily decreased from 0.147 → 0.109 over 10 epochs, confirming the model was genuinely _learning_ to ignore color rather than just being regularized.

#### Method C: Color-Adversarial Head (Gradient Reversal Layer)

Attach a second classification head that predicts background color from the CNN's feature representation. A gradient reversal layer (GRL) flips the sign of gradients from the color head, so the encoder is _rewarded_ for making the color head fail:

```
L = L_CE(digit) + γ · L_CE(color, with gradient reversal)
```

Results were humbling:

| γ       | Test Hard | Train | Color Acc | Verdict                                |
| ------- | --------- | ----- | --------- | -------------------------------------- |
| **0.1** | **38.6%** | 96.8% | 96.7%     | Modest gain, but color head still wins |
| 0.5     | 27.7%     | 96.0% | 96.0%     | Barely above baseline                  |
| 1.0     | 21.3%     | 83.4% | 83.8%     | Training destabilized                  |
| 2.0     | 11.4%     | 11.1% | 11.1%     | Complete collapse — random guessing    |

**Key diagnostic:** color classification accuracy **never dropped toward chance** (10%). At γ=0.1, it actually _climbed_ from 83% → 97% during training. The GRL was too weak to strip color from the features. With only 4,486 parameters, there isn't enough representational capacity for a meaningful adversarial game.

#### Method D: Combined (Best of Both)

Using the best settings from each (α=2.0 + γ=0.1) trained for 15 epochs. Reached **61.3%** on test-hard — strong, but actually _slightly worse_ than jitter alone (62.5%). If jitter already makes color unreliable, there's not much color signal left for the adversary to fight over.

### Results

| Method                   | Train | Val   | Test Hard | Gap  |
| ------------------------ | ----- | ----- | --------- | ---- |
| Baseline                 | 96.1% | 95.5% | 22.7%     | 0.73 |
| Color-Jitter (α=2.0)     | 95.4% | 95.2% | **62.5%** | 0.33 |
| Adversarial (γ=0.1)      | 96.8% | 96.5% | 38.6%     | 0.58 |
| Combined (α=2.0 + γ=0.1) | 95.9% | 96.0% | 61.3%     | 0.35 |

- **Hard Test Accuracy (Lazy baseline):** 22.7%
- **Hard Test Accuracy (Best robust model):** 62.5% (Color-Jitter α=2.0)

The winner is color-jitter consistency alone — simple, stable, and the most effective. The adversarial head adds theoretical elegance but not practical value, at least for a network this small. The best model is saved as `outputs/best_intervention_model.pt`.

Grad-CAM comparison between baseline and intervention model confirmed the shift: baseline heatmaps had hot spots on the background, while the jitter-trained model had hot spots on the digit strokes.

### Sources I referenced for Task 4

- https://arxiv.org/pdf/1907.02893
- https://core.ac.uk/outputs/647998231/?source=2
- https://arxiv.org/abs/2305.12686
- https://arxiv.org/abs/2310.13977
- https://arxiv.org/abs/2404.05058

---

## 7. Task 5 — The Invisible Cloak (Targeted attack)

### Goal

Targeted adversarial attack:

- Start with a digit **7**
- Add perturbation δ so the model predicts **3**
- Confidence **> 90%**
- Constraint: **max pixel change ε < 0.05**
- Compare robust vs lazy model by required noise magnitude.

### What I implemented

I implemented a **Projected Gradient Descent (PGD)** targeted attack with:

- **Random start** inside the ε-ball (Uniform(-ε, ε) initialization — true PGD, not iterative FGSM)
- Iterative gradient ascent on the target class log-probability
- L∞ projection: clamp perturbation to [-ε, +ε] per pixel after each step
- Pixel clamp: keep the adversarial image in valid [0, 1] range
- Step size α = ε / 10, up to 1000 optimization steps
- Both models use raw [0, 1] RGB tensors (no normalization anywhere in the pipeline), so the comparison is apples-to-apples

### Comparison and quantification

I compared:

- how often each model can be forced to predict 3 with **ε = 0.05**
- or the minimum ε required to hit **>90% confidence**

**Results:**

- Lazy model: ✅ SUCCESS at step 108 — target confidence 90.0%, L∞ = 0.050, L2 = 2.28
- Robust model: ✅ SUCCESS at step 78 — target confidence 90.2%, L∞ = 0.050, L2 = 2.20
- Minimum ε for lazy model to reach 90%: **0.0505**
- Minimum ε for robust model to reach 90%: **0.0476**

**Multi-image attack (N=20 random 7s):**

| Metric                | Lazy        | Robust          |
| --------------------- | ----------- | --------------- |
| Success rate          | 14/20 (70%) | **19/20 (95%)** |
| Avg target confidence | 88.3%       | 90.2%           |
| Avg steps to success  | 161.0       | **38.5**        |
| Avg L2 perturbation   | 2.07        | **1.83**        |

**The surprise:** the robust model is _easier_ to fool adversarially, not harder. The headline scalar:

```
ε_min(robust) / ε_min(lazy) ≈ 0.94
```

Distribution-shift robustness ≠ adversarial robustness. Color-jitter training smoothed the loss landscape, making gradients more consistent and predictable — exactly what PGD thrives on. The lazy model's fragile, chaotic loss surface (latched onto a color shortcut) is actually harder for PGD's fixed-step gradient ascent to navigate.

### Sources used for Task 5

- https://github.com/Harry24k/adversarial-attacks-pytorch?utm_source=chatgpt.com
- https://arxiv.org/abs/1706.06083?utm_source=chatgpt.com
- https://docs.pytorch.org/tutorials/beginner/fgsm_tutorial.html?utm_source=chatgpt.com