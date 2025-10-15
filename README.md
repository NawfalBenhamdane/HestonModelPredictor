# DeepHeston  
### Neural Estimation of the Heston Stochastic Volatility Model Parameters

**Author:** Nawfal Benhamdane  
**Machine Learning & AI Student**  
📧 _[nawfal.benhamdane@student-cs.fr]_  
🌐 [LinkedIn](https://linkedin.com/in/nawfal-benhamdane-6298b1285//) | [GitHub](https://github.com/NawfalBenhamdane)

---

## Project Overview

**DeepHeston** is a research project in **machine learning applied to quantitative finance**.  
It aims to estimate the **parameters of the Heston stochastic volatility model** using a **deep neural network** architecture combining **bidirectional LSTM networks** and a **multi-head attention mechanism**.

The model learns to calibrate the latent parameters of the Heston model directly from financial time-series data (prices, volatility, and returns) without explicit supervision.

---

## Objective

The main objective is to **replace classical numerical calibration methods** (which are often unstable and computationally expensive) with a **differentiable neural calibration** approach.

> Predict the latent parameters *(κ, θ, σᵥ, ρ, µ)* of the Heston model from historical financial time-series.

These parameters can then be used to **simulate realistic market dynamics** for pricing, forecasting, or stress testing.

---

## The Heston Model

The Heston model (1993) describes the joint evolution of an asset price *(Sₜ)* and its instantaneous variance *(vₜ)* through the following stochastic differential equations:

$$
\begin{aligned}
dS_t &= \mu S_t \, dt + \sqrt{v_t} \, S_t \, dW_t^{(1)} \\
dv_t &= \kappa (\theta - v_t) \, dt + \sigma_v \sqrt{v_t} \, dW_t^{(2)}
\end{aligned}
$$

with

$$
E[dW_t^{(1)} dW_t^{(2)}] = \rho \, dt
$$

The model captures **stochastic volatility**, **price-volatility correlation** (leverage effect), and reproduces the **volatility smile** observed in financial markets.

---

## Dataset

- **Source:** [Yahoo Finance via yFinance](https://pypi.org/project/yfinance/)  
- **Asset studied:** S&P 500 Index (^GSPC)  
- **Time span:** 1928 → 2025  
- **Frequency:** Daily (trading days only)

### Features
- `Close`: Adjusted closing price  
- `Volatility`: Historical or implied volatility (via VIX)  
- `Returns`: Log-transformed daily returns  

**Preprocessing:**
- Removed missing values (due to rolling window)  
- Chronological sorting and index reset  
- Standardization (z-score normalization)  
- 30-day rolling input windows  

---

## Model Architecture

The architecture follows an **Encoder–Decoder** structure:

1. **Temporal Encoder**  
   - Three stacked **bidirectional LSTM** layers  
   - Captures long-term temporal dependencies  

2. **Multi-Head Attention Module**  
   - Dynamically weights important time steps  
   - Detects volatility regimes and trend shifts  

3. **Weighted Temporal Pooling**  
   - Aggregates recent time information with increasing temporal weights  

4. **Dense Prediction Head (MLP)**  
   - Outputs the five Heston parameters  
   - Uses `softplus` and `tanh` to enforce physical constraints  

---

## Loss Function & Training

The total loss combines several components:

$$
L_{total} = L_{price} + L_{vol} + \lambda_{feller} L_{feller} + \lambda_{reg} L_{reg}
$$

- **Lₚᵣᵢ𝒸ₑ**: Relative error between simulated and observed prices  
- **Lᵥₒₗ**: Relative error between simulated and observed volatility  
- **L_feller**: Penalty enforcing the Feller condition  


- **L_reg**: L2 regularization on the predicted parameters  

### Key Hyperparameters
| Parameter | Value |
|------------|--------|
| Sequence length | 30 days |
| LSTM hidden size | 256 |
| Attention heads | 8 |
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| λ_feller | 0.1 |
| λ_reg | 0.001 |

---

## Experimental Results

### Prediction Quality
The model accurately reproduces short-term market dynamics for both **price** and **volatility**, showing strong coherence between simulated and observed data.

### Parameter Distribution
| Parameter | Typical Range | Interpretation |
|------------|----------------|----------------|
| κ | [0.5, 4.0] | Mean-reversion speed |
| θ | [0.02, 0.12] | Long-term variance level |
| σᵥ | [0.1, 0.7] | Volatility of variance |
| ρ | [-0.99, -0.2] | Negative leverage effect |
| µ | ≈ 0.02 | Average annual growth rate |

### Free Simulation
When run in **autonomous predictive mode**, the model can simulate realistic market trajectories over 30–200 days, maintaining plausible variance dynamics and price evolution.

---

## Discussion

**DeepHeston** demonstrates the feasibility of a **neural calibration framework** for stochastic volatility models.  
It offers a stable, differentiable, and theoretically consistent alternative to traditional numerical methods.

Potential applications include:
- Option pricing  
- Stress testing  
- Scenario generation  
- Probabilistic volatility forecasting  
