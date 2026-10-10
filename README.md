# scikit-learn Cookbook

**Code reproduction + theoretical deep-dive of _scikit-learn Cookbook, Third Edition_** (John Sukup, O'Reilly / Packt).

This repository is my submission for **Task 2 (Enrichment for Machine Learning Classes): "Code Reproduction + Theoretical Deep-Dive from scikit-learn Cookbook (O'Reilly)"**.
For every chapter of the book there is one Jupyter notebook that (1) reproduces the book's code, (2) summarises the chapter and explains the theory behind each recipe, and (3) adds small experiments that put numbers behind the book's statements.


---

## Repository structure

```
scikit-learn-Cookbook/
├── README.md
├── 01-common-conventions-and-api-elements/
│   └── ch01_sklearn_api.ipynb
├── 02-pre-model-workflow-and-data-preprocessing/
│   └── ch02_data_preprocessing.ipynb
├── 03-dimensionality-reduction-techniques/
│   └── ch03_dimensionality_reduction.ipynb
├── 04-distance-metrics-and-nearest-neighbors/
│   └── ch04_knn_distance_metrics.ipynb
├── 05-linear-models-and-regularization/
│   └── ch05_linear_models_regularization.ipynb
├── 06-advanced-logistic-regression-and-extensions/
│   └── ch06_logistic_regression_extensions.ipynb
├── 07-support-vector-machines-and-kernel-methods/
│   └── ch07_svm_kernel_methods.ipynb
└── 08-tree-based-algorithms-and-ensemble-methods/
    └── ch08_tree_ensemble_methods.ipynb
```

## How each notebook is organised

Every notebook follows the book's recipes **in the same order** and uses the same structure:

| Part | What it contains |
|------|------------------|
| **Theory** | the idea behind the recipe, with the key formulas (written in LaTeX) |
| **Reproduction** | the book's code, re-run, with outputs saved in the notebook |
| **Extra experiments** | small, measurable checks, for example verifying a formula by hand or comparing methods with repeated cross-validation |
| **📝 Note** | places where the book is imprecise, out of date for current scikit-learn, or cannot be reproduced exactly, with the reason |
| **Chapter summary** | a summary table, key takeaways and self-check questions (collapsible answers) |

The explanations are paraphrased in my own words. All numbers quoted in the text were checked against the outputs stored in the notebooks.

## Getting started

**Requirements:** Python 3 and scikit-learn **1.6 or newer** (Chapter 1 uses `validate_data` and estimator tags, which appeared in 1.6). The notebooks were executed with **scikit-learn 1.8.0**; each notebook prints its library versions in the first cell.

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn jupyter
jupyter lab
```

Notes on running:

- All datasets are generated locally or bundled with scikit-learn, so everything runs **offline**, with one exception: the closing exercise of Chapter 2 uses `fetch_california_housing()`, which downloads the data on first use. Those five cells have **no stored output**; run them once with internet access.
- `seaborn` is only needed for the heatmaps in Chapter 4.
- Approximate run time (fresh run, one machine): Chapter 3 ≈ 1.5 minutes, Chapter 4 ≈ 30 seconds, Chapter 5 ≈ 2.5 minutes (mostly t-SNE and the repeated cross-validation loops), Chapter 7 ≈ 1 minute (grid searches). Chapters 1, 2, 6 and 8 are quick.
- Random seeds are fixed in every notebook (2024 in Chapters 2-4 and 6-8, as in the book; 123 in Chapter 5), so results are reproducible. Small numeric differences can still appear across library versions.

---

## Chapter summaries

### Chapter 1: Common Conventions and API Elements of scikit-learn

**In general.** Before introducing any algorithm, the book explains the shared "grammar" of the library. Every object speaks the same small vocabulary, so a random forest, a scaler and a clustering model feel like variations of one tool, and swapping one algorithm for another is a one-line change.

**Topics covered (9 recipes):** design philosophy (consistency, simplicity, modularity, reusability) · estimators (`fit`, `predict`, `fit_predict`) · transformers (`fit`, `transform`, `fit_transform`) · custom estimators and transformers (`BaseEstimator` + mixins) · pipelines and a note on MLOps · common attributes and methods (`coef_`, `intercept_`, `score`) · hyperparameter tuning (`get_params` / `set_params`, grid, random and halving search) · metadata (estimator tags and metadata routing) · best practices.

**What the notebook adds**
- Four very different classifiers trained with the *identical* loop body, to show what "consistent API" means in practice.
- A demonstration that re-fitting a scaler on the test set hides distribution shift (the correctly scaled test mean stays about +1.8 standard deviations away from zero; the wrong way forces it back to 0).
- Two custom components written from scratch (a quantile winsorizer and a mean-baseline regressor) that plug into `Pipeline` and `GridSearchCV`.
- Pipeline persistence with `joblib`, metadata routing with `sample_weight` (including the error scikit-learn raises when the request is not declared), and a small tags table.

**Key idea:** _fit on training data, transform everything else_, and let a `Pipeline` enforce it.

---

### Chapter 2: Pre-Model Workflow and Data Preprocessing

**In general.** "Garbage in, garbage out": no algorithm can rescue data that is incomplete, badly scaled or encoded in a form the model cannot read. The chapter covers the cleaning steps that come before modelling and shows why they should be packaged in pipelines.

**Topics covered:** the impact of raw data on model performance · handling missing data (`SimpleImputer`, `KNNImputer`, `IterativeImputer`) · scaling (`StandardScaler`, `MinMaxScaler`, `Normalizer`, plus robust/power/quantile variants) · encoding categorical variables (`OneHotEncoder`, `OrdinalEncoder`, `LabelEncoder`, `ColumnTransformer`) · pipelines and data leakage · feature engineering (`PolynomialFeatures`, `KBinsDiscretizer`, `RFE`, `SelectFromModel`) · a practical exercise on California Housing · a decision guide for choosing a scaler.

**What the notebook adds**
- An imputation experiment on complete data with 15 % of the cells hidden: with correlated features, `KNNImputer` and `IterativeImputer` reduce the error by roughly 40-50 % compared with the mean; with independent features they offer no gain.
- Scaling measured on the Wine dataset: k-NN improves from about 0.69 to 0.95 and SVM from about 0.66 to 0.98, while a random forest is unaffected.
- A leakage experiment on pure noise: selecting features *before* cross-validation reports about 84 % accuracy, while the correct in-pipeline version stays at chance.
- An end-to-end pipeline with missing values in numeric and categorical columns, tuned with `GridSearchCV`.

**Corrections to the book:** `KNNImputer` averages the neighbours' values (it does not take a "majority label"); the "> 5 %" guideline for `SimpleImputer` is most likely a typo for "< 5 %"; scaling is merely *unnecessary* for trees, not harmful; `LabelEncoder` numbers categories alphabetically and is meant for targets (use `OrdinalEncoder` for ordered features); California Housing has 8 features, not 9.

---

### Chapter 3: Dimensionality Reduction Techniques

**In general.** Having more features is not automatically better: too many redundant or noisy features make data hard to visualise, slow to process and easy to over-fit. The chapter presents three techniques that answer three different questions: **PCA** (which directions carry the most variance?), **LDA** (which directions separate the classes best?) and **t-SNE** (how can high-dimensional data be drawn so that neighbours stay neighbours?).

**Topics covered:** why reduce dimensions, selection versus extraction · PCA · LDA · t-SNE · a decision flow for choosing a technique · the effect of reduction on model performance and two practical exercises.

**What the notebook adds**
- The curse of dimensionality measured: the relative contrast between the farthest and nearest pair of random points falls from hundreds in 2-D to 0.17 in 1000-D.
- PCA verified by hand against an eigen-decomposition of the covariance matrix; scree plot, biplot and reconstruction error. On Wine, two components keep 55.41 % of the variance, and 10 are needed for 95 %.
- Why scaling matters: without it, PC1 "explains" 99.8 % of the variance simply because `proline` has the largest numbers.
- LDA with two components matches a model that uses all 13 Wine features (0.988 vs 0.980), while PCA with two components is clearly lower (0.964). LDA fitted outside the pipeline reports 100 % accuracy on pure noise.
- t-SNE quantified with trustworthiness (about 0.99, against 0.82 for PCA and 0.77 for LDA on Digits) and its limits: no `transform()`, sensitivity to perplexity and seed.
- On Digits, 40 of 64 PCA components are as accurate as the full model (0.969 vs 0.972); on Iris, two components keep 96 % of the variance yet still lose some accuracy.

**Corrections to the book:** the arrows in the book's PCA plot show the weights of the first two original features, not the directions of PC1 and PC2 (a proper biplot is provided); `load_digits()` is the small UCI dataset (1 797 images of 8×8 pixels), not MNIST.

---

### Chapter 4: Building Models with Distance Metrics and Nearest Neighbors

**In general.** The chapter turns the human notion of "similar" into mathematics with a distance, and builds one of the simplest models on top of it: k-nearest neighbours (KNN), a lazy learner that stores the data and decides each prediction by a vote (or an average) of the closest points.

**Topics covered:** distance metrics (Euclidean, Manhattan, Minkowski, and cosine, Hamming, Jaccard) · how KNN works and the role of *k* · comparing metrics · hyperparameter tuning with `GridSearchCV` and `RandomizedSearchCV` · evaluation (cross-validation, learning curve, confusion matrix, classification report) · three practical exercises · `RadiusNeighborsClassifier`.

**What the notebook adds**
- A from-scratch reproduction of a single KNN prediction, decision boundaries for several values of *k*, and KNN for regression.
- Repeating the metric comparison 100-200 times: Manhattan's advantage on the checkerboard data averages only 0.6 accuracy points, so the single-split gap in the book is mostly luck.
- Scale sensitivity: Wine improves from about 0.70 to 0.96 with standardisation, a single huge-scale noise column drops Iris from 0.963 to 0.372 (0.927 after scaling), and Digits (all pixels already on one scale) is slightly *worse* after scaling (0.986 → 0.976).
- Irrelevant features hurt KNN far more than logistic regression: with 200 noise columns KNN falls to about 0.71, logistic regression to about 0.90.
- Nested cross-validation shows that `best_score_` from a grid search is optimistic (0.975 vs 0.961 honest).
- On a 95 % / 5 % imbalanced problem, accuracy is 0.967 while always predicting the majority class already gives 0.947, and minority recall is only 0.38.
- Exercises: the scaler lifts a KNN on Wine from 0.667 to 0.978 test accuracy; a grid search on Digits chooses no scaler, the cosine metric and *k* = 3, reaching 0.984 test accuracy (7 mistakes in 450 images).

**Corrections to the book:** `metric="minkowski"` with the default `p=2` *is* the Euclidean distance, so two of the three metrics compared in Figure 4.2 are identical; the dataset called "moons" is a checkerboard; `iris.data` and `iris.target` are attributes, not methods; the book's `make_circles` call has no seed, so its numbers cannot be reproduced exactly.

---

### Chapter 5: Linear Models and Regularization

**In general.** The chapter starts with ordinary least squares (OLS) and asks what goes wrong when a line has too much freedom: many features, correlated features or few samples. The answer is regularization, a penalty on the size of the coefficients: **Ridge** (L2) shrinks them, **Lasso** (L1) shrinks some to exactly zero, and **ElasticNet** mixes both. The chapter closes with polynomial and spline regression for non-linear data.

**Topics covered:** linear models and OLS · Ridge and Lasso · ElasticNet · model complexity, bias-variance and Bayesian/group-penalty pointers · polynomial regression and splines · three practical exercises.

**What the notebook adds**
- Theory verified numerically: the normal equations equal `LinearRegression`; with orthonormal features Lasso equals soft-thresholding of the OLS coefficients and Ridge equals OLS divided by a constant; the geometry of the L1 diamond versus the L2 disc.
- **A diagnosis of the book's main dataset.** The loop that makes features correlated overwrites 7 of the 10 informative features, destroying about 77 % of the signal, so no linear model can exceed an R² of roughly 0.23. The book's R² values (0.09-0.14) should be read against that ceiling.
- A corrected dataset (150 samples, 50 informative-plus-noise features with noisy copies): test MSE of 2 761 for OLS, 939 for Ridge and 680 for Lasso, against a noise floor of 400. With more features than samples, OLS and Ridge are no better than the mean while Lasso reaches R² 0.993 and finds all 5 true features.
- ElasticNet's apparent win in the book comes from stronger regularization: a Ridge with `alpha=400` gives the same cross-validated R² (0.1828).
- Bias-variance measured by simulation, readable coefficient paths, the grouping effect shown with bootstrap resamples, and BayesianRidge.
- Polynomials versus splines: degree 5 (the book's "best") has a cross-validated MSE of 47.6, degree 7 reaches 15.9, and even degrees add nothing because the target is an odd function. Outside the data range the polynomials miss the truth by roughly 7 times while splines stay bounded.
- In the closing exercises a single 80/20 split makes plain OLS look fine, but repeated cross-validation reveals its instability (standard deviation 1.26) and shows tuned Lasso at about 13 % above the noise floor.

**Corrections to the book:** the `alpha` of Ridge and of Lasso/ElasticNet are on different scales (the latter divide the error by 2n), so they cannot be compared; the book's ElasticNet notation uses constants that scikit-learn does not have; the "spline interpolation" is really a regression spline; the correlated noise in the dataset is unseeded, so the book's exact numbers vary from run to run.

---

### Chapter 6: Advanced Logistic Regression and Extensions

**In general.** Logistic regression is introduced properly here: not as a tool for predicting numbers, but for predicting class probabilities through the sigmoid function. The chapter then stretches the same idea in three directions — more than two classes, a penalty on the coefficients, and more than one label per sample — before closing with the metrics needed to judge whether any of it actually worked.

**Topics covered (6 recipes):** overview of logistic regression and the sigmoid/log-odds link · multiclass classification (One-vs-Rest and multinomial/softmax) · regularization (Ridge L2, Lasso L1) · multilabel classification · model evaluation metrics · scikit-learn implementation notes.

**What the notebook adds**
- Plain logistic regression on Breast Cancer reaches 94.7 % accuracy out of the box; precision/recall/F1 are reported alongside accuracy to show why a single number is not enough for a medical-screening use case.
- OvR and multinomial compared on Iris: 95.6 % vs 93.3 % accuracy — close enough to show that, on a well-separated dataset, the two multiclass strategies matter less than they would on harder data.
- Ridge vs Lasso on Breast Cancer, both at `C=0.1`: Ridge keeps all 30 features and reaches 98.2 % accuracy; Lasso zeroes out 23 of the 30 coefficients (keeps only 7) and still reaches 96.5 %, visualised with coefficient-path plots across eight `C` values.
- A multilabel classifier (one `LogisticRegression` per label, binary-relevance style) on a synthetic 5-label dataset: exact-match accuracy is only 34.0 %, far below the per-label F1-scores, which is used to show why exact-match is a harsh metric when several independent labels must all be right at once. A label co-occurrence heatmap is added on top of the book's code.
- ROC/AUC on Breast Cancer: AUC ≈ 0.989, with a short "which metric when" guide (imbalanced data, ranking tasks, screening vs spam-filtering) that the book does not spell out explicitly.

**📝 Note (scikit-learn compatibility):** the book builds OvR with `LogisticRegression(solver='liblinear')` directly. In current scikit-learn, `liblinear` no longer fits multiclass problems on its own and the `multi_class` parameter has been removed (multinomial/softmax is now the implicit default for ≥ 3 classes), so the notebook wraps the OvR model explicitly in `OneVsRestClassifier` to keep the recipe meaningful.

---

### Chapter 7: Support Vector Machines and Kernel Methods

**In general.** Earlier chapters mostly used data that is easy to separate. This chapter admits that real boundaries are rarely clean and introduces SVMs: find the hyperplane with the widest margin, and if the data is not linearly separable, use a kernel to make it look as if it were — in a higher-dimensional space.

**Topics covered (5 recipes):** introduction to SVMs and the hyperplane/margin/support-vector idea · kernel functions (linear, polynomial, RBF) and the kernel trick · tuning SVM parameters (`C`, `degree`, `gamma`) with grid search · SVMs in high-dimensional ("wide") data · evaluating SVM models.

**What the notebook adds**
- A linear-kernel SVM on Iris reaches 93.3 % accuracy; an RBF-kernel `SVR` regression example is included alongside the classifier (MSE ≈ 0.204) so both SVM tasks from the book are covered side by side.
- Linear, polynomial and RBF kernels compared on the same split, plus a 2-feature decision-boundary plot for all three kernels in one figure, to make the (sometimes subtle) differences between kernels visible rather than just numeric.
- Grid search over `C`, `kernel` and `degree`: best cross-validation score 0.990, but test accuracy lands back at 0.933 — used to point out that a higher CV score does not automatically mean a higher held-out score, especially on a small, easy dataset where many configurations are close together.
- A `C`-only sweep (log-spaced, eight values) plotted against cross-validation score, showing a clear plateau: past a certain `C`, more regularization strength buys nothing.
- A synthetic 1000-feature, 1000-sample dataset (`make_classification`) to test SVM in a genuinely wide regime: default-parameter accuracy is only 74.7 %, visibly weaker than on Iris, with a PCA projection to 2-D showing the two classes heavily overlapping at that reduced dimensionality.
- ROC/AUC on Breast Cancer: AUC ≈ 0.984, noting that `probability=True` is required for `predict_proba` and makes fitting slightly slower (scikit-learn runs an internal calibration pass).

---

### Chapter 8: Tree-Based Algorithms and Ensemble Methods

**In general.** Not every useful algorithm needs heavy mathematics. A decision tree just keeps splitting the data to make each resulting group purer, and is easy to read top to bottom. The chapter's real point, though, is that a single tree overfits easily, and that combining many of them — in parallel (bagging) or in sequence (boosting) — is usually worth the extra complexity.

**Topics covered (5 recipes):** introduction to decision trees (Gini impurity, `max_depth`, `min_samples_split`/`leaf`, `max_features`) · random forests and bagging · gradient boosting machines · hyperparameter tuning for trees and ensembles · comparing ensemble methods (bagging vs boosting vs stacking).

**What the notebook adds**
- A single decision tree on Iris reaches 86.7 % accuracy; the full tree diagram is plotted and read node by node (split feature, Gini, sample proportion, majority class) rather than just quoting the final score.
- Random forest (100 trees) lifts the same split to 91.1 % accuracy, with a feature-importance bar chart confirming petal length/width as the dominant features — consistent with what the single tree already implied.
- Gradient boosting (1000 estimators, `learning_rate=0.2`) lands at exactly 86.7 %, matching the single tree rather than beating it — flagged explicitly as a sign that both models may be hitting the same performance ceiling on a dataset this simple, not that boosting "failed".
- A 27-combination grid search over `n_estimators` × `max_depth` × `learning_rate` for GBM: best parameters give the same 86.7 % test accuracy as the untuned tree, with a heatmap of cross-validation scores showing a clear plateau once depth and estimator count pass a modest threshold.
- Bagging, boosting and stacking (`StackingClassifier` with a logistic-regression meta-model) compared directly: random forest (91.1 %) narrowly beats both GBM and stacking (86.7 % each) on this split — used to make the point that a fancier ensemble (stacking) does not automatically beat a simpler one; it depends on how much the base models' errors actually complement each other.

---

## Lessons that repeat across chapters

1. **Fit on training data only, then transform everything else.** Put every data-dependent step (imputing, scaling, feature selection, LDA, PCA) inside a `Pipeline`; otherwise evaluation scores are inflated (Chapters 1-3).
2. **Scale before distance- or penalty-based models** (k-NN, SVM, PCA, Ridge/Lasso), but not out of habit: trees and tree-based ensembles do not need it, and Digits pixels are better left alone (Chapters 2-5, 7-8).
3. **Do not trust a single split.** Differences of one or two test samples were noise in several chapters; repeated cross-validation told the real story (Chapters 3-5).
4. **Tune hyperparameters with cross-validation, and report an honest score.** `best_score_` is optimistic; a hand-picked `alpha` or *k* is an untuned hyperparameter (Chapters 1, 4, 5). A higher CV score during grid search does not always translate into a higher held-out test score, especially on small, easy datasets (Chapters 7-8).
5. **Check the data and the claim before the conclusion.** Several results in the book (a "best" model, an "improvement") shrink or vanish once the dataset, the scale of a parameter or the comparison is examined (Chapters 4-5).
6. **Retaining variance, accuracy or R² is not the same as retaining signal**: always compare with a baseline and with the noise level.
7. **More model complexity is not automatically better.** Multinomial vs OvR, more boosting rounds, or a stacked meta-model all showed diminishing or even flat returns once a simple baseline was already strong (Chapters 6-8).

## Reference

Sukup, J. _scikit-learn Cookbook, Third Edition: Over 80 recipes for machine learning in Python with scikit-learn._ O'Reilly / Packt.
