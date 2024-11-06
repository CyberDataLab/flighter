# ALGORITHMS.md

## Overview of Algorithms Used

Our manuscript introduces sophisticated algorithms within the Defense and Communications Modules to ensure the security, integrity, and efficiency of federated learning conducted by a fleet of aerial vehicles. These algorithms address the challenges posed by adversarial behavior and communication latency in a dynamic environment. This document outlines the methodologies and mathematical formulations used.

---

## Defense Module: Key Components and Methodologies

The Defense Module mitigates the risk posed by malicious nodes attempting to introduce poisoned models or engage in adversarial behaviors. This module ensures the integrity and security of model updates through a multi-step process.

### 1. Model Similarity

The module continuously monitors the similarity of model updates, comparing them to the aggregated model from previous rounds and other participants' updates. Any significant deviation from the expected learning trajectory can indicate a potential poisoning attack.

#### Cosine Similarity

One of the methods used to measure the similarity between model updates is cosine similarity. The cosine similarity between two model update vectors $\mathbf{u}$ and $\mathbf{v}$ is defined as:

$$
\mathrm{cosine\_sim}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}
$$

where:
- $\mathbf{u} \cdot \mathbf{v}$ is the dot product of the vectors
- $\|\mathbf{u}\|$ and $\|\mathbf{v}\|$ are the magnitudes (Euclidean norms) of the vectors

By calculating the cosine similarity, the module can detect deviations in the direction of model updates, which may indicate potential adversarial behavior.

### 2. Situational Awareness Metrics

The module evaluates the operational integrity of each aerial vehicle using the following situational awareness metrics:
- **Flight Formation**: Ensures vehicles maintain their designated positions within the formation, monitoring deviations from the standardized scheme.
- **Geopositioning**: Real-time geographical location data is analyzed using the Euclidean distance between a vehicle's position and its planned trajectory.
- **Interconnection**: Communication latency is monitored to detect network disruptions, such as dropped or delayed connections.
- **Resource Usage**: Tracks computational resources to identify anomalies, such as unusual CPU or memory usage.

### 3. Defense Score Calculation

The Defense Score $\mathcal{S}_i(t)$ for each aerial vehicle $i$ at federated round $t$ is computed as:

$$
\mathcal{S}_i(t) = \int_0^t \Phi\left( \mathbf{w}(t) \cdot \mathbf{M}_i(\tau) - \Omega_i(\tau) \right) d\tau
$$

#### Weight Adjustments

The dynamic weights $\mathbf{w}(t)$ are defined as:

$$
\mathbf{w}(t) = \left( w_{MS}(t), w_{FF}(t), w_{GP}(t), w_{IC}(t), w_{RU}(t) \right)
$$

$$
\mathbf{w}(t) = \left( 0.30 \cdot e^{-\alpha t}, 0.30 \cdot e^{-\beta t}, 0.15 \cdot e^{-\gamma t}, 0.15 \cdot e^{-\delta t}, 0.10 \cdot e^{-\epsilon t} \right)
$$

### 4. Regularization Term

The regularization term $\Omega_i(t)$ penalizes deviations from expected behavior:

$$
\Omega_i(t) = \sum_{k=1}^6 \sigma_k^2 \cdot \left( \mathcal{M}_{i,k}(t) - \mathbb{E}[\mathcal{M}_k(t)] \right)^2
$$

where:
- $\sigma_k^2$: Variance of the $k$-th metric
- $\mathcal{M}_{i,k}(t)$: Observed value of the $k$-th metric
- $\mathbb{E}[\mathcal{M}_k(t)]$: Expected value across all vehicles

### 5. Dynamic Threshold

The Defense Score is compared to a time-evolving threshold $\Theta(t)$:

$$
\Theta(t) = \frac{0.9}{1 + e^{-k(t - 0.2T)}}
$$

### 6. Classification of Aerial Vehicles

Vehicles are classified as benign or malicious based on:

$$
\text{Class}(i,t) =
\begin{cases} 
  \text{benign}, & \mathcal{S}_i(t) \geq \Theta(t) \\
  \text{malicious}, & \mathcal{S}_i(t) < \Theta(t)
\end{cases}
$$

---

## Communications Module: Latency and Reliability Modeling

The Communications Module ensures efficient and reliable communication between aerial vehicles, accounting for dynamic mobility and network conditions.

### 1. Communication Latency Model

Communication latency $L_{ij}(t)$ between vehicles $i$ and $j$ at time $t$ is calculated as:

$$
L_{ij}(t) = \frac{D_{ij}(t)}{c} + \Delta_{\text{processing}} + \Delta_{\text{queuing}}
$$

where:
- $D_{ij}(t)$: Distance between vehicles $i$ and $j$ at time $t$
- $c$: Propagation speed of the communication signal
- $\Delta_{\text{processing}}$: Processing delays at the sender and receiver
- $\Delta_{\text{queuing}}$: Queuing delays due to network congestion

#### Distance Calculation

The distance $D_{ij}(t)$ is derived from the position vectors:

$$
D_{ij}(t) = \left\| \mathbf{p}_i(t) - \mathbf{p}_j(t) \right\|
$$

where $\mathbf{p}_i(t)$ and $\mathbf{p}_j(t)$ are the position vectors of vehicles $i$ and $j$ at time $t$. The positions update based on velocities:

$$
\mathbf{p}_i(t + \Delta t) = \mathbf{p}_i(t) + \mathbf{v}_i(t) \Delta t
$$

### 2. Integrating Latency into the Defense Module

To address mobility-induced communication challenges, we link the decay rate $\gamma$ in $w_{GP}(t)$ to the average latency $L_i(t)$:

$$
\gamma_i(t) = \gamma_0 + \kappa L_i(t)
$$

where:
- $\gamma_0$: Base decay rate
- $\kappa$: Proportionality constant reflecting latency sensitivity
- $L_i(t)$: Average communication latency for vehicle $i$:

$$
L_i(t) = \frac{1}{N - 1} \sum_{j \neq i} L_{ij}(t)
$$

### 3. Adjusted Weight Expression

The weight $w_{GP}(t)$ is dynamically adjusted:

$$
w_{GP}(t) = 0.15 \cdot e^{-(\gamma_0 + \kappa L_i(t)) t}
$$

This adjustment reduces the influence of the Geopositioning metric when communication conditions deteriorate.

### 4. Communication Reliability

Communication link reliability $R_{ij}(t)$ is modeled as:

$$
R_{ij}(t) = e^{-\lambda L_{ij}(t)} \cdot P_{\text{link}}(t)
$$

where:
- $\lambda$: Decay constant for reliability decrease with latency
- $P_{\text{link}}(t)$: Probability of link stability, influenced by signal-to-noise ratio and environmental factors
