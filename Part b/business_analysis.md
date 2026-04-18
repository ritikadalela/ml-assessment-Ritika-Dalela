# Part B: Business Case Analysis

## B1. Framing the Business Problem

### (a) How to cast this as a machine learning task
This can be framed as a **supervised learning regression problem**. The outcome the retailer wants to estimate is **`items_sold`** for a given store during a given monthly promotion period.

Possible predictor variables would include:
- store-level attributes such as `store_id`, `store_size`, `location_type`, customer profile indicators, and local competition density
- campaign-related inputs such as `promotion_type`, promotional depth, category focus, loyalty mechanics, and campaign duration
- calendar signals such as month, season, holiday periods, festivals, and weekend intensity
- recent performance variables such as lagged sales, prior response to promotions, and rolling trends in store demand

Regression is the correct choice because the model needs to predict a **continuous numeric value** rather than a class label. After generating predicted `items_sold` for each possible promotion, the business can choose the option with the strongest expected outcome for each store-month combination.

### (b) Why `items_sold` is preferable to total sales revenue
Using `items_sold` is more appropriate because it directly reflects the operational objective: increasing unit movement through the stores. Revenue can move up or down for reasons that do not necessarily indicate promotional success, such as price changes, discount depth, premium product mix, or inflationary effects.

For instance, one promotion may generate strong product movement but lower revenue per unit because it relies on heavier discounts. Another may produce higher revenue simply because it concentrates demand in expensive categories, even if it does not actually improve promotional effectiveness in a comparable way.

The broader lesson is that the target variable must be tightly aligned with the **real decision the business wants to improve**. A model can look good on paper, but if the target does not reflect the true business objective, the recommendations may still be misleading.

### (c) Better alternative to a single global model
Instead of fitting one pooled model across all 50 stores, I would recommend a **segmented or hierarchical modelling strategy**. One practical design would be:
- a global model to learn broad retailer-wide relationships, and
- additional models by location segment, store cluster, or region to capture local response patterns.

This is justified because customer behaviour, competitive context, and promotion sensitivity are unlikely to be uniform across urban, semi-urban, and rural stores. A single model may average away those differences, while a segmented approach can preserve meaningful local variation.

## B2. Data Design and Exploratory Analysis

### (a) Combining the raw tables and defining the modelling grain
I would first decide on the grain of the modelling table. The most sensible structure is:

**one row = one store for one month under one promotion setup**

The joins would be handled as follows:
- **transactions** would be aggregated from raw transaction level up to store-month level
- **store attributes** would be merged using `store_id`
- **promotion details** would be merged using the promotion identifier plus the relevant time key
- **calendar data** would be merged using the date or month key to add festival, holiday, and weekend-related indicators

Before training, I would aggregate the transaction data to match the chosen unit of analysis. Typical measures created at this stage could include:
- total `items_sold`
- transaction count
- average basket size or value
- average selling price
- category-level sales mix
- lagged demand indicators from earlier months

The final dataset should therefore be a clean analytical table where each row represents one store-month decision context and every feature is available at prediction time.

### (b) EDA plan before modelling
Before building the model, I would carry out several checks and visual analyses.

**1. Distribution analysis for the target and numeric drivers**  
I would inspect histograms and boxplots for `items_sold`, competition density, footfall, and similar numeric fields. I would be looking for skewness, outliers, unusual spikes, or impossible values. These findings would help determine whether transformations, clipping, or robust modelling choices are needed.

**2. Promotion performance across store segments**  
I would compare `items_sold` by `promotion_type`, split by `location_type`, `store_size`, or store clusters. Boxplots or grouped bar charts would help reveal whether some promotions work better in certain segments than others. If strong interaction patterns appear, I would explicitly model interactions or move toward segment-specific models.

**3. Seasonality and calendar effects**  
I would chart monthly sales trends, festival months, and weekend-heavy periods. The goal would be to see whether demand changes systematically over time. If clear seasonal effects exist, I would engineer temporal features such as month, quarter, holiday windows, and lag terms.

**4. Correlation and multicollinearity checks**  
A correlation heatmap and summary statistics would help identify highly related numeric features. This matters because overlapping variables can distort interpretation and inflate instability in simpler models. If strong redundancy appears, I would reduce or combine features.

**5. Missing-data review**  
I would profile missingness by table, feature, and store segment. This helps distinguish true zero activity from absent records and informs whether to use imputation, exclusion, or missing-indicator features.

**6. Store-level heterogeneity review**  
I would compare store-level averages and variability in outcomes. If certain stores consistently behave differently, that would support adding store identifiers, store embeddings, clustering, or multi-level modelling.

### (c) Handling the fact that 80% of transactions have no promotion
If most observations involve no promotion, the model may learn to favour the no-promotion pattern simply because it dominates the sample. That can reduce its ability to distinguish the effect of individual campaigns and may bias predictions toward business-as-usual behaviour.

To address this, I would:
- create a clear indicator for whether a promotion is active
- evaluate performance separately on promoted and non-promoted periods
- consider reweighting or stratified validation so promotion periods are properly represented
- examine uplift or incremental performance rather than relying only on overall average error
- ensure each promotion type has enough coverage before treating its recommendations as reliable

## B3. Evaluation, Interpretation, and Deployment

### (a) Train-test design and metrics
Because the data is monthly and time-ordered over three years, I would use a **chronological split**, not a random one. For example:
- use the earliest period for training,
- reserve a later block for validation,
- and hold out the most recent months as the final test set.

A random split is inappropriate because it leaks future information into the training sample. In production, the model must forecast future store-month outcomes using only information available from the past, so the evaluation setup needs to mirror that reality.

The core evaluation metrics I would use are:
- **MAE**: gives the average absolute error in units sold and is easy for business teams to interpret
- **RMSE**: penalises large misses more heavily, which is useful when badly misallocating a promotion has a high business cost
- **MAPE or WAPE** where appropriate: useful for understanding relative error across stores of different scale
- **Segment-level error breakdowns** by store type, region, and promotion type so we can check whether the model is strong overall but weak in certain business segments

In practice, MAE tells the business how far off the predictions are on average, while RMSE highlights whether the model occasionally makes very costly errors.

### (b) Explaining why the same store gets different promotion recommendations in different months
If the model recommends Loyalty Points Bonus for Store 12 in December but Flat Discount for the same store in March, I would investigate the features that changed between those two prediction contexts.

Feature importance can help show which variables most influence the model overall, such as seasonality, festival indicators, prior store performance, promotion type, or local competition. I would then pair that global view with a local explanation for the two specific predictions, showing how December differs from March in terms of seasonal demand, customer behaviour, event periods, and store conditions.

When communicating this to the marketing team, I would avoid only saying “the model picked a different promotion.” Instead, I would explain the business drivers in plain language, for example:
- December may coincide with festive demand and higher basket potential, making loyalty-based promotions more effective
- March may behave more like a routine demand period where direct price incentives generate stronger volume

That style of explanation makes the recommendation easier to trust and act on.

### (c) End-to-end deployment process
Once the model is trained, I would save the fitted preprocessing-and-model pipeline using a serialisation approach such as `joblib` or `pickle`. Saving the full pipeline is important so that the exact same feature transformations are applied at scoring time.

At the start of each month, a new scoring dataset would be prepared by:
1. pulling the latest store attributes, promotion candidates, and calendar information
2. generating the same engineered features used during training
3. passing that prepared dataset into the saved pipeline
4. producing predicted `items_sold` for each candidate promotion-store combination
5. selecting the recommendation with the strongest predicted outcome

The output could then be written to a dashboard, a planning table, or an automated recommendation file for the marketing team.

Monitoring should be built into the deployment workflow. I would track:
- prediction error over time once actual outcomes become available
- drift in important input features such as promotion mix, store traffic, and competition
- segment-level performance by location type, store size, and promotion type
- unusual changes in the distribution of recommendations

If those monitoring checks show meaningful degradation, I would trigger retraining using newer data so the model remains aligned with current customer and market behaviour.
