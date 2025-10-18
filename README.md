## INSTALL:
```
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
./unzip.sh
```

## USAGE:

### Quick Start

```bash
# Run with default settings (Panel model with Kalman smoothing and CV)
python run_models.py

# Run OLS model
python run_models.py --model-type ols

# Run without cross-validation
python run_models.py --no-cv
```

## Command Line Arguments

### Model Selection

#### `--model-type {ols, panel}`
- **Type:** Choice
- **Default:** `panel`
- **Description:** Select the type of model to run
- **Options:**
  - `ols`: Ordinary Least Squares regression
  - `panel`: Panel OLS with fixed effects

**Example:**
```bash
python run_models.py --model-type ols
```

---

#### `--model-name MODEL_NAME`
- **Type:** String
- **Default:** `None` (runs all models)
- **Description:** Run a specific model configuration by name
- **Note:** Model names must exist in `models.py` (for Panel) or `OLS_models` (for OLS)

**Example:**
```bash
python run_models.py --model-name limousine_model1
```

---

### Data Processing Options

#### `--kalman` / `--no-kalman`
- **Type:** Flag
- **Default:** `True` (Kalman smoothing enabled)
- **Description:** Apply Kalman smoothing to weight measurements to reduce measurement noise

**Examples:**
```bash
# Enable Kalman smoothing (default)
python run_models.py --kalman

# Disable Kalman smoothing
python run_models.py --no-kalman
```

---

#### `--measurement-noise NOISE`
- **Type:** Float
- **Default:** `400`
- **Description:** Expected measurement error variance for Kalman filter
- **Note:** Only relevant when Kalman smoothing is enabled

**Example:**
```bash
python run_models.py --measurement-noise 500
```

---

#### `--cut-tails`
- **Type:** Flag
- **Default:** `False`
- **Description:** Remove the bottom 2.5% and top 2.5% of data based on `pred_adgLatest_average` to eliminate outliers

**Example:**
```bash
python run_models.py --cut-tails
```

---

#### `--n-weighings N [N ...]`
- **Type:** Integer list
- **Default:** `[1]`
- **Description:** List of prediction horizons (number of weighings to use)

**Examples:**
```bash
# Single horizon
python run_models.py --n-weighings 7

# Multiple horizons
python run_models.py --n-weighings 1 7 14 21
```

---

### Cross-Validation Options 

#### `--cv` / `--no-cv`
- **Type:** Flag
- **Default:** `True` (CV enabled)
- **Description:** Enable or disable k-fold cross-validation for Panel models
- **Note:** Only applies to Panel models; ignored for OLS models

**Examples:**
```bash
# Enable cross-validation (default)
python run_models.py --model-type panel --cv

# Disable cross-validation
python run_models.py --model-type panel --no-cv
```

---

#### `--k-folds K`
- **Type:** Integer
- **Default:** `5`
- **Description:** Number of folds for cross-validation
- **Note:** Only relevant when CV is enabled

**Example:**
```bash
python run_models.py --k-folds 10
```


### Model Types

#### OLS (Ordinary Least Squares)

- **Description:** Standard linear regression
- **Best for:** Simple relationships, quick analysis
- **Features:**
  - No fixed effects
  - Faster computation
  - Assumes independence between observations


---

#### Panel OLS (Fixed Effects)

- **Description:** Panel regression with entity (cow) fixed effects
- **Best for:** Repeated measures data, controlling for individual differences
- **Features:**
  - Controls for cow-specific effects
  - Accounts for within-cow variation over time
  - More robust for panel data structure
  - Supports cross-validation with GroupKFold

### Console Output

The tool provides detailed console output including:
1. **Configuration Summary:** Shows all selected options
2. **Data Processing:** Reports on smoothing and filtering operations
3. **Model Training:** Progress through each model and dataset
4. **Model Summary:** Statistical results for each model
5. **Cross-Validation Results:** Mean and standard deviation of metrics (if CV enabled)
6. **Diagnostics:** Residual analysis and model fit statistics


### Saved Files

For each model run, the following files are saved:

**JSON Results:**
```
model_results/
└── {model_name}/
    └── {n}_results.json
```

Contains:
- Model coefficients
- Standard errors
- P-values
- R², MAE, RMSE metrics
- Cross-validation results (if applicable)
- Diagnostics

**Plots:**
```
model_results/
└── PanelOLS_{model_name}/
    └── {n}_plot.png
```

Shows:
- Predicted vs Actual scatter plot
- Model fit metrics
- CV metrics (if applicable)
---


### Appendix: Complete Argument Reference

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model-type` | choice | `panel` | Model type: `ols` or `panel` |
| `--model-name` | str | `None` | Specific model to run |
| `--kalman` / `--no-kalman` | flag | `True` | Enable/disable Kalman smoothing |
| `--measurement-noise` | float | `400` | Kalman filter measurement noise |
| `--cut-tails` | flag | `False` | Remove bottom/top 2.5% outliers |
| `--n-weighings` | int[] | `[1]` | Prediction horizons |
| `--cv` / `--no-cv` | flag | `True` | Enable/disable CV (Panel only) |
| `--k-folds` | int | `5` | Number of CV folds |

---

*Last Updated: October 2025*

## NOTES:
Medical history was incorrectly recorded, so data about that is unreliable.
Cow with ID: rexFmUY8QHCvB0TsjnbB had major issues so is left out of the analysis.

## NOTES:
Neg required = a * weight + adg^2
Neg required (Negr) = a * weight + (Neg/Negr)^2
Neg required (Negr) = a * weight + Neg^2/Negr^2
Neg required (Negr) = a * weight + Neg^2/Negr^2

0 = a * weight + Neg^2/Negr^2 - Negr 

which gives approx:

negr = a * weight neg^2(a * weight)^2

or if weight is small

Negr = Neg^2/3

sinds average pref on the cow farm is 1 adg -> negr = neg per day

would require some sort of hidden state model.

for now Negr is approx 1, -> more advanced methods need more data.


## TODO:

explain the breed split.
explain all the finding with the ratio, and the dmi/dt etc. processes/
explain different values for Negr that failed etc.
make all the in between models so it can easily be checked
make 3 data processing options -> raw, kalhman filter, kahlman filter + tail removal.



Try to come up with a generalized from of wg over long periods.

show difference between OLS and Panel

Conculsions


