# 🧠 DeepHeston  
### Estimation neuronale des paramètres du modèle stochastique de Heston

**Auteur :** Nawfal Benhamdane  
**Machine Learning & AI Student — Fraud Detection and Generative Modeling.**  
📧 _[nawfal.benhamdane@student-cs.fr]_  
🌐 [LinkedIn](https://linkedin.com/in/nawfal-benhamdane-6298b1285//) | [GitHub](https://github.com/NawfalBenhamdane)

---

## 📘 Description du projet

**DeepHeston** est un projet de recherche en **machine learning appliqué à la finance quantitative**.  
Il vise à estimer les **paramètres du modèle stochastique de Heston** à l’aide d’une architecture **neuronale profonde**, combinant **réseaux LSTM bidirectionnels** et **mécanismes d’attention multi-têtes**.

Le modèle apprend à calibrer les paramètres latents du modèle de Heston directement à partir de séries temporelles de données financières (prix, volatilité, rendements), sans supervision explicite.

---

## 🎯 Objectif

L’objectif principal est de **remplacer la calibration numérique classique** (souvent instable et coûteuse) par une **calibration neuronale différentiable** :

> Prédire les paramètres latents *(κ, θ, σᵥ, ρ, µ)* du modèle de Heston à partir d’une séquence temporelle d’observations de marché.

Ces paramètres permettent ensuite de **simuler la dynamique du marché** (prix & volatilité) de manière réaliste.

---

## 🧩 Fondements théoriques

### Le modèle de Heston

Le modèle de Heston (1993) décrit l’évolution conjointe du prix d’un actif *(Sₜ)* et de sa variance instantanée *(vₜ)* via le système d’équations stochastiques :

\[
\begin{cases}
dS_t = \mu S_t dt + \sqrt{v_t} S_t dW^{(1)}_t \\
dv_t = \kappa (\theta - v_t) dt + \sigma_v \sqrt{v_t} dW^{(2)}_t
\end{cases}
\]

avec \( E[dW^{(1)} dW^{(2)}] = \rho dt \).

Le modèle capture la **volatilité stochastique**, la **corrélation prix-volatilité** (effet levier), et reproduit le **smile de volatilité** observé sur les marchés.

---

## 📊 Données utilisées

- **Source :** [Yahoo Finance via yFinance](https://pypi.org/project/yfinance/)  
- **Actif étudié :** Indice S&P 500 (^GSPC)  
- **Période couverte :** 1928 → 2025  
- **Fréquence :** Journalière (jours ouvrés)

### Variables utilisées :
- `Close` : Prix de clôture ajusté  
- `Volatility` : Volatilité historique ou implicite (via VIX)  
- `Returns` : Rendement log-transformé  

Prétraitement :
- Suppression des valeurs manquantes  
- Normalisation (centrage-réduction)  
- Fenêtrage temporel sur 30 jours glissants  

---

## 🧠 Architecture du modèle

L’architecture suit une logique **Encoder–Decoder** :

1. **Encodeur temporel** :
   - 3 couches **LSTM bidirectionnelles**
   - Extraction du contexte passé/futur des séquences financières  

2. **Mécanisme d’attention multi-têtes** :
   - Pondération dynamique des jours les plus informatifs  
   - Capture des ruptures de tendance et régimes de volatilité  

3. **Pooling temporel pondéré** :
   - Accent sur les observations récentes  

4. **Prédicteur dense (MLP)** :
   - Sortie : les 5 paramètres du modèle de Heston  
   - Contraintes physiques assurées via fonctions `softplus` et `tanh`  

---

## ⚙️ Fonction de perte & Entraînement

La fonction de perte totale combine plusieurs composantes :

\[
L_{total} = L_{prix} + L_{vol} + \lambda_{feller} L_{feller} + \lambda_{reg} L_{reg}
\]

- **Lprix** : Erreur sur le prix simulé  
- **Lvol** : Erreur sur la variance simulée  
- **Lfeller** : Pénalité sur la condition de Feller \( 2\kappa\theta > \sigma_v^2 \)  
- **Lreg** : Régularisation L2  

### Hyperparamètres clés
| Paramètre | Valeur |
|------------|--------|
| Longueur de séquence | 30 jours |
| Taille cachée (LSTM) | 256 |
| Nombre de têtes d’attention | 8 |
| Optimiseur | AdamW |
| Taux d’apprentissage | 1e-3 |
| λ_feller | 0.1 |
| λ_reg | 0.001 |

---

## 📈 Résultats expérimentaux

### 🔹 Qualité des prédictions
Le modèle reproduit fidèlement la dynamique réelle du marché à court terme, tant sur les **prix** que sur la **volatilité**.

### 🔹 Distribution des paramètres estimés
| Paramètre | Intervalle typique | Interprétation |
|------------|--------------------|----------------|
| κ | [0.5, 4.0] | Retour à la moyenne modéré à fort |
| θ | [0.02, 0.12] | Variance cible réaliste |
| σᵥ | [0.1, 0.7] | Volatilité de la variance maîtrisée |
| ρ | [-0.99, -0.2] | Effet levier négatif |
| µ | ≈ 0.02 | Croissance moyenne annuelle |

### 🔹 Simulation libre
En mode **prédictif autonome**, le modèle peut simuler la trajectoire future du prix et de la variance sur 30 à 200 jours, générant des scénarios réalistes et cohérents avec la structure des marchés.

---

## 💬 Discussion

L’approche **DeepHeston** démontre la **faisabilité d’une calibration neuronale** stable et cohérente du modèle de Heston, contournant les limites des méthodes numériques classiques.  
Elle ouvre la voie à des applications en :
- **Pricing d’options**
- **Stress testing**
- **Génération de scénarios de marché**
- **Prévision probabiliste de la volatilité**
