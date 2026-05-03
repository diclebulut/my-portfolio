
# Dicle Bulut - Data Science & Machine Learning Portfolio
 
# Summary

I'm a data scientist with hands-on experience in dynamic pricing, demand forecasting, and production ML systems. Alongside my professional work delivering £3M+ in business value through machine learning, optimisation, and AI solutions, I've built these projects to deepen expertise in specific domains: pricing strategy, forecasting, NLP, and geospatial analysis. 

---

# Projects

## 1. Dynamic Pricing with Reinforcement Learning - Uber NYC Data

**Repository:** [dynamic-pricing-uber-data](https://github.com/diclebulut/dynamic-pricing-uber-data) 
**Stack:** Python, PyTorch, scikit-learn, RL

### The Context
Ride-sharing platforms operate in a dynamic environment where 
demand, supply, and customer behaviour fluctuate constantly. Understanding how different pricing strategies perform against each other is critical for competitive advantage in this space.

### The Problem
Traditional flat-rate pricing leaves money on the table during high-demand periods and turns away customers during low-demand windows. How can dynamic pricing adapt in real-time to optimize revenue while maintaining acceptable acceptance rates?

### The Solution
Built a **Deep Q-Network (DQN) reinforcement learning agent** that learns optimal price multipliers (0.8x–1.2x) from real Uber trip data, trained against a realistic **sigmoid-based acceptance model** that captures willingness-to-pay dynamics.

### The Approach
**Data:** Real NYC Uber data from 2021, incorporating practical factors like trip distance, weather conditions, time of day, neighbourhoods, and customer behaviour signals (Source: Kaggle)

**Static Approach:** 
- Learns one optimal price-per-mile from historical data and applies it uniformly across all rides.
- Models customer acceptance using a price sensitivity curve: the more you deviate from the base fare, the fewer customers accept.

**Dynamic Approach:**
- Learns context-dependent price multipliers (0.8x–1.2x) for each trip using a 4-layer neural network trained on trip features via experience replay and epsilon-greedy exploration. 
- Selects the best multiplier from 11 options based on what maximises profit for that specific trip context.
- Uses the identical sigmoid-based price-sensitivity curve, but implicitly learns which trips tolerate premium pricing

### Model Choice

**Alternative candidates considered:**
- **Contextual bandits (LinUCB / Thompson Sampling):** strong for one-step price selection and often more sample-efficient, but primarily optimizes immediate reward and does not naturally extend to sequential effects (e.g., demand feedback, churn, driver supply response).
- **Supervised demand modelling + optimisation (e.g., XGBoost to estimate P(accept|x, price) then choose argmax profit):** interpretable and a strong tabular baseline, but depends on having sufficient historical price variation to learn counterfactual price sensitivity reliably.
- **Continuous-control RL (DDPG/TD3/SAC) or actor–critic methods (PPO):** enable fully continuous price multipliers, but add substantial complexity and tuning sensitivity compared to discrete-action value learning.

**Choosing DQN - Reasoning:**
- **RL baseline:** DQN is a standard, well-understood method with a relatively straightforward setup. It’s a practical way to test whether an RL policy can beat static pricing without over-engineering the first version.
- **Lower complexity than continuous-action RL:** Methods like SAC/TD3/DDPG are strong choices for truly continuous pricing, but they introduce extra moving parts (actor–critic design, exploration noise, more tuning). For this project, DQN reached a working, comparable result faster.
- **Next steps:** If this were taken further, the same setup could be extended to continuous multipliers (SAC/TD3).

### Technical Highlights

**Willingness-to-Pay Modelling:**
- Designed a sigmoid-based price acceptance model that maps the absolute deviation between the quoted price and the trip’s baseline (“actual”) fare into an acceptance probability 
- Calibrated the curve from training data to reflect real-world acceptance thresholds
- The agent learns that aggressive pricing kills acceptance; conservative pricing leaves revenue on the table

<img width="618" height="176" alt="image" src="https://github.com/user-attachments/assets/7cc49dba-f59d-4872-a2d7-200d1f5ed83a" />

Where:
- **f:** “actual” fare for the trip
- **p_hat:** predicted price for the trip (in the DQN env this is computed as p_hat = f  multiplier)
- **delta (abs(p_hat - f)):** absolute deviation between quote and baseline
- **r:** max acceptable deviation ratio (default 0.5)
- **delta_max (r*f):** maximum “acceptable” deviation from the actual fare
- **s:** sigmoid scaling factor ratio, controls how quickly the acceptance probability drops as proposed price gets farther away from the actual fare (default 0.1)


**DQN Architecture & Training:**
- **4-layer neural network incl. output layer** (256 → 256 → 128 → 11 actions) with experience replay (2,000-sample buffer)
- **11 discrete price actions** representing multipliers across the 0.8x–1.2x range
- **ε-greedy exploration** with exponential decay (ε₀=1, decay=0.995), forcing systematic exploration early, converging to exploitation
- **Target network updates** to stabilize Q-value estimation

### Comments Received

Feedback from a former Uber pricing practitioner indicated that the project’s willingness-to-pay / acceptance modelling is broadly consistent with industry approaches used in ride-hailing pricing, and that the core intuition around the price–demand tradeoff matches real-world practice.

### Data Ingestion & Pipeline Architecture

**Data Sources:**
- **Primary data:** Uber NYC trip records (Parquet format) from January 2021
- **Weather data:** NYC weather CSV with precipitation measurements and types
- **Geospatial reference:** NYC Taxi Zone lookup mapping LocationIDs to boroughs

**Ingestion & Cleaning:**
- Filter to Uber-specific trips (`hvfhs_license_num == "HV0003"`)
- Remove invalid records (trip_miles > 0)
- Multi-source merge on temporal (pickup date) and spatial (location IDs) keys

**Feature Engineering & Domain Logic:**
- **Trip characteristics:** distance (miles), tips, wait time (seconds between request and pickup), day of month
- **Financial metrics:** final_fare (total customer payment including tolls, taxes, surcharges), uber_profit, profit-per-mile, profit-per-second, driver_final_pay
- **Geospatial features:** Pickup_Borough and Dropoff_Borough encoding from taxi zone lookup (Manhattan vs outer boroughs have different congestion profiles)
- **Temporal features:** time_of_day binning (5 periods capturing rush hour vs. off-peak dynamics: night/late/morning/midday/evening)
- **Weather integration:** precipitation amount and type merged on pickup date (affects both demand and acceptance); missing values filled with 'None'
- **Behavioral proxy:** wait time (seconds between request and pickup)
- **Demand signals:** shared_request_flag and shared_match_flag (proxy for demand intensity at the moment of quote)
- **Dispatch indicators:** out_of_base_dispatch_flag (cross-base assignment signal)

**Model Preparation Pipeline:**
- **Train/test split:** 80/20 stratification with fixed random seed for reproducibility
- **Numerical features** (5): trip_miles, tips, wait, day_of_month, precip - StandardScaler normalization (fitted on training data only to prevent leakage)
- **Categorical features** (6): boroughs, preciptype, time_of_day, shared flags - One-hot encoding with `drop='first'` to avoid multicollinearity
- **Target variable:** final_fare (amount customer pays)

**Architecture Highlights:**
- **Modular design:** Separate modules for preprocessing, static pricing, and RL components
- **Configuration-driven:** All paths, features, and hyperparameters centralized in `config.py`
- **Reproducibility:** Fixed seeds across NumPy, PyTorch, and Python's random library
- **Computational scope:** January 1st subset (10,000 rides) for demonstration purposes


### Outcomes of DQN Compared to Static
| **Metric** | **Static** | **DQN** | **Change** |
|----------|----------|----------|----------|
| Total Revenue | $32,316 | $47,608 | +47.32%|
| Acceptance Rate | 81.75% | 99.70% | +17.95pp |
| Avg Price per Ride| $23.46| $23.88 | +1.8% |
| Accepted Rides | 1,635 | 1,994 | +359 rides |

### Next Steps
- SAC vs DQN implementation.
- Reward function: It is aggressive, it cuts prices off at x1.2 multiplier. A continuous curve can be applied.
- The model tries to assign a passenger fare to each state. Reward function could be made more complex with profit and cost information.
- Features in reward function: another model can be layered on top of this to give the agent more information about how to reward presence of some features e.g. rain.
- More test consistency by 1-to-1 match between static and DQN test sets.
- Applying this approach to the driver side pay offering.

---

## 2. Flight Delay Prediction - U.S. Domestic Flights 2015

**Repository:** [flight-delays](https://github.com/diclebulut/flight-delays) 
**Stack:** Python, scikit-learn, pandas, GridSearchCV

### The Problem
Airlines must forecast departure delays to optimize operational reliability, gate allocation, and crew scheduling. With 5.8M flights in the dataset and complex interdependencies, statistical forecasting is essential.

### The Solution
Built a **forecasting pipeline** with domain-driven feature engineering, systematically comparing Logistic Regression, Random Forest, and Gradient Boosting. **Random Forest** emerges as the best performer with **AUC 0.73** and CV score **0.74** after hyperparameter tuning and cross-validation.

### Model Choice

**Alternative candidates considered:**
- **Regularised Logistic Regression (baseline):** fast to train, easy to explain. Limitation: it misses non-linear effects and interactions (e.g., airport × airline × time-of-day patterns).
- **Gradient Boosting (XGBoost / LightGBM / CatBoost):** frequently the strongest family for tabular prediction because it captures interactions effectively. Tradeoff: higher tuning effort and greater sensitivity to hyperparameters.
- **Neural networks (MLP):** an MLP (multi-layer perceptron) is a feed-forward neural network for tabular features. It can model non-linearities, but typically requires time-consuming regularisation and tuning to match strong tree ensembles on structured data, and is harder to interpret.
- **Time-series / forecasting approaches:** useful where delays are driven mainly by temporal dependence (e.g., knock-on effects during the day). Limitation: flight delay risk is also strongly context-dependent (route, carrier, airport, calendar features), so time-series methods alone may miss that.
- **Regression formulation (predict minutes delayed):** provides a richer output than classification, but the target can be heavy-tailed and noisy, and it complicates threshold-based decisions such as “delay > 15 minutes”, which is a standard used across airlines.

**Choosing Random Forest - Reasoning:**
- **Captures non-linear structure with minimal assumptions:** it models interactions between categorical and numerical drivers without requiring a strictly linear relationship between features and delay risk.
- **Strong tabular baseline:** Random Forest is a reliable benchmark for structured data and performed well under the same cross-validation setup used for model comparison.
- **Suitable for mixed feature types after encoding:** it can utilise the combination of calendar variables, route/airport signals, and airline/aircraft indicators effectively.

**Operational Relevance and Metrics:**
In operational planning (staffing, gate allocation, turnaround buffers), the practical value often depends on the chosen decision threshold and the resulting precision/recall tradeoff, not only AUC.

**Time/compute caveat:**
Random Forest can be computationally expensive at this scale, and iteration time becomes a constraint when running repeated cross-validation and tuning. Gradient Boosting remains an important alternative for this reason.

### Technical Highlights
 
**Model Development:**
- **Stratified 5-fold cross-validation:** stratified by the target class (delayed/not-delayed), Airline balance is preserved separately via stratified sampling at the data preparation stage
- **Three-model comparison:** Logistic Regression (baseline interpretability), Random Forest (non-linearity), Gradient Boosting (best performance)
- **GridSearchCV tuning:** 27 candidate configurations × 3-fold inner CV

**Production Considerations:**
- **Config-driven pipeline**: feature lists, thresholds, file paths centralized in `config.py`
- **Run logging**: every training serialized to JSON with timestamps, hyperparameters, and metrics for experiment tracking and audit trail
- **Model persistence**: timestamped `joblib` serialization for versioning

### Data Ingestion & Pipeline Architecture

**Multi-Source Data Integration:**
- **Flight records:** 5.8M U.S. domestic flights (2015) from DOT Bureau of Transportation Statistics
- **Airport metadata:** IATA codes, lat/lon coordinates, city/state mapping for origin/destination enrichment
- **Aircraft reference:** Tail number - FAA model mapping from SFO registry 
- **Geospatial merge:** Left joins on origin and destination airports to enrich each flight with departure/arrival location attributes

**Data Cleaning & Sampling:**
- **Exclusions:** Remove cancelled and diverted flights (focus on completed operations only)
- **Stratified sampling by airline:** 5% sample per carrier to preserve airline-specific delay patterns while managing 5.8M record scale
- **Time format standardization:** Convert HHMM integer format (e.g., 1430) to HH:MM strings, then parse to extract hour/minute features

**Feature Engineering & Domain Logic:**

- **Temporal features:**
  - Time-of-day extraction (hour/minute) from 6 schedule/actual timestamps (departure, arrival, wheels-off, wheels-on)
  - Calendar features: quarter, week-of-year, IS_WEEKEND flag
  - Business intelligence: `IS_BUSINESS_FLIGHT` (Monday 6-9 AM), `IS_HOLIDAY_SEASON` (summer, winter, July 4th)

- **Geospatial features:**
  - Origin/destination state, city, lat/lon (regional congestion proxy)
  - `IS_LEVEL_3_AIRPORT`: JFK, LGA, DCA slot-controlled facilities (capacity constraints)

- **Aircraft characteristics:**
  - Tail number - model mapping (turnaround time, gate compatibility, mechanical risk profiles)
  - Top-50 categorical encoding for aircraft models (rare models grouped as "Other")

- **Target engineering:**
  - Binary delay flag: 1 if delay > 15 minutes (industry-standard threshold), else 0

**Model Preparation Pipeline:**
- **Train/test split:** 80/20 stratified by target (preserves delay rate across splits); airline balance maintained through upstream stratified sampling
- **ColumnTransformer architecture:**
  - **Numerical features** (9): distance, hour/minute extracts, day, week - median imputation + StandardScaler
  - **Categorical features** (7): airline_top, origin_top, destination_top, state_dep_top, state_arr_top, aircraft_model_top - most-frequent imputation + one-hot encoding (handle_unknown='ignore' for production resilience)

### Outcomes
**Model Comparison:**
- Logistic Regression: AUC 0.6958 (CV: 0.6892 ± 0.0306)
- Random Forest: AUC 0.7284 (CV: 0.7360 ± 0.0181) ✓ **Selected**
- Gradient Boosting: AUC 0.7432 (CV: 0.7315 ± 0.0263)

**Random Forest Performance:**
- Test AUC: 0.7284
- Precision (delayed): 0.64 | Recall (delayed): 0.18
- Overall accuracy: 0.83
- Top predictive signals (aggregated): STATE (19.9%), DAY (16.4%), DESTINATION (12.0%), ORIGIN (11.3%), AIRCRAFT (7.9%), SCHEDULED_DEPARTURE_HOUR (7.5%)

---

## 3. NLP Text Classification for CBI Economic Intelligence

**Repository:** [Primary-Topic-Classifier-Model](https://github.com/diclebulut/Random-Forest-Insights-Primary-Topic-Classifier-Model-with-Supervised-Machine-Learning) 
**Stack:** Python, spaCy, NLTK, scikit-learn, LexVec embeddings

### The Problem
CBI staff manually read and categorised thousands of business survey anecdotes into 5 topics (Demand Impact, People, Policy, Supply, Other). Manual classification was slow (hours per batch), subjective, and unscalable.

### The Solution
End-to-end NLP pipeline using **LexVec word embeddings (2M words, 300-dim) + Random Forest classifier**, achieving **~90% reduction in processing time**.

### Technical Highlights

**NLP Preprocessing:**
- **Tokenisation & normalisation** via spaCy (`en_core_web_sm`); lowercasing and filtering
- **Domain-aware stopword removal** removing NLTK stopwords with tuning (e.g., retain "people" as it signals the People topic)

**Word Embedding & Sentence Vectorisation:**
- **Pre-trained LexVec CommonCrawl embeddings** (300 dimensions), 2M vocabulary
- **Sentence-level representation:** Average word vectors - single 300-dim vector per anecdote
  - Simple yet effective; avoids LSTM/Transformer complexity
  - Pre-computed vectors serialised via `pickle`

**Model Training & Validation:**
- **Random Forest (100 estimators)**
- **Stratified 5-fold cross-validation**: preserves class distribution (ensures all 5 topics represented in train/test folds)
- **Hyperparameter grid search:** `max_depth ∈ {None, 15, 12, 9, 6}` × `min_samples_leaf ∈ {1, 2, 4, 8, 16, 32}` = 30 configurations × 5 folds
- **Overfitting detection:** Compared training vs. test accuracy per fold

**Output & Operationalisation:**
Note: This section was taken out of the public repository due to IP limitations. Summary:
- **Enriched Excel exports**: original anecdote + cleaned tokens + vector coordinates + human labels + model predictions
- **Probability distribution**: `predict_proba()` outputs score for each topic; allows ranking confidence and flagging uncertain cases for human review
- **Reusable pipeline**: new survey batches could be processed by re-running vectorisation

---

## 4. Earthquake Prediction - Turkey Seismic Analysis (Work In Progress!)

**Repository:** [earthquake-prediction](https://github.com/diclebulut/earthquake-prediction) | **Stack:** Python, pandas, SciPy, Folium, geospatial analysis, Markov chains (planned)

### The Problem
Earthquakes cluster along known fault lines, but their timing and magnitude are hard to predict. Can historical earthquake + fault geological data reveal temporal and spatial patterns for probabilistic forecasting?

### The Solution
Multi-source data pipeline integrating **Kandilli Observatory earthquake XML feeds** with **GEM Global Active Faults Database (GeoJSON)**, implementing sophisticated **spatial-temporal matching** and building foundation for **Markov chain-based forecasting**.

### Technical Highlights

**Data Architecture:**
- **Earthquake data ingestion:** HTTP download of monthly XML feeds from Boğaziçi University; parsed into structured DataFrames with magnitude (ML scale), lat/lon, depth, location
- **Fault database:** GeoJSON from GEM, includes geological properties: dip angle, rake, slip type, seismogenic depth, net slip rate
- **Bounding-box filtering:** Query fault database within to only include faults around the events in scope

**Fault–Earthquake Matching Engine:**
- **Spatial proximity matching:** For each earthquake, find closest fault using **SciPy `cdist`
- **Multi-radius binning:** Classify earthquakes by proximity to faults: 0–5 km, 5–10 km, 10–20 km, 25-50 
- **Property merge:** Enrich earthquake records with fault characteristics (slip rate, dip angle, etc.) for downstream modelling

**Analytical Outputs:**
- **Per-fault statistics:** Event counts, magnitude distributions, temporal trends for each fault
- **Interactive visualisation:** Folium maps with magnitude-coloured markers, fault overlays, clustering, per-catalog toggles


| Map                           | Tooltip                          |
| ----------------------------------- | ----------------------------------- |
| <img width="550" height="464" alt="image" src="https://github.com/user-attachments/assets/042b1442-4525-4713-9b83-8597e508f4af" />| <img width="556" height="468" alt="image" src="https://github.com/user-attachments/assets/51621812-9746-4727-8a4e-93602869d414" />|


**Foundation for Markov Chain Model:**
A Markov Chain Model is planned for the prediction section of this project for the following reasons:

- **State discretisation:** Each fault occupies a state (Locked, Creeping, Critically Stressed, Ruptured) based on recent seismic activity
- **Transition probabilities:** Estimated from historical catalogue; weighted by recency (recent earthquakes have stronger influence on future probability)
- **Cumulative feedback:** After each earthquake, fault state updates, reflecting stress accumulation/release cycle
- **Spatial coupling (future):** Extended model to account for stress transfer between neighbouring faults (would require the implementation of inter-fault relations)

### Domain Insights Leveraged
1. Earthquakes near a fault alter probability of subsequent events on same fault
2. More recent earthquakes dominate future predictions (temporal recency weighting)
3. Stress can cascade to neighbouring faults (spatial Markov chains)
4. Fault properties (dip, slip type, rake) influence rupture dynamics
5. Magnitude scale is logarithmic (10× amplitude per unit)




