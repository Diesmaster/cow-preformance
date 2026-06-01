data {
  int<lower=1> N;
  int<lower=1> n_animals;

  array[N] int<lower=1>         animal_id;
  vector[N]                     obs_weight;
  vector[N]                     days_gap;
  vector[N]                     next_gap;
  array[N] int<lower=0,upper=1> is_first;
  array[N] int<lower=0,upper=1> is_last;
  array[N] int<lower=1>         prev_idx;
  array[N] int<lower=1>         next_idx;

  vector[N] obs_tdn_silage;
  vector[N] obs_tdn_tahu;
  vector[N] obs_tdn_SP2A;
  vector[N] obs_tdn_SMG;
  vector[N] obs_tdn_rumput;

  vector[N] hasBEF_dt;
  vector[N] hormone_effect_dt;
  vector[N] gotDewormed_dt;
  vector[N] gotHNMVaccination_dt;
}

parameters {
  // --- non-centered latent weight innovations ---
  // standard normal, reconstructed into true_weight in transformed parameters
  vector[N] weight_raw;

  // --- initial weight ---
  real          mu_init;
  real<lower=0> sigma_init;

  // --- weight noise ---
  real<lower=0> sigma_obs;
  real<lower=0> sigma_process;

  // --- growth rate (non-centered) ---
  real<lower=0>     mu_growth;
  real<lower=0>     sigma_growth;
  vector[n_animals] growth_rate_z;

  // --- feed consumption rates ---
  real<lower=0, upper=1> rate_silage;
  real<lower=0, upper=1> rate_tahu;
  real<lower=0, upper=1> rate_SP2A;
  real<lower=0, upper=1> rate_SMG;
  real<lower=0, upper=1> rate_rumput;

  // --- maintenance modifiers ---
  real beta_BEF;
  real beta_hormone;
  real beta_dewormed;
  real beta_vaccination;

  // --- energy to growth conversion ---
  real<lower=0> beta_NEg;
}

transformed parameters {
  // --- growth rates ---
  vector[n_animals] growth_rate = mu_growth + sigma_growth * growth_rate_z;

  // --- reconstruct true_weight from innovations (non-centered) ---
  // this is what makes HMC able to explore the latent state space
  vector[N] true_weight;
  for (n in 1:N) {
    if (is_first[n] == 1) {
      true_weight[n] = mu_init + sigma_init * weight_raw[n];
    } else {
      int a = animal_id[n];
      true_weight[n] = true_weight[prev_idx[n]]
                     + growth_rate[a] * days_gap[n]
                     + sigma_process * sqrt(days_gap[n]) * weight_raw[n];
    }
  }

  // --- metabolic weight ---
  vector[N] metabolic_weight;
  for (n in 1:N) {
    metabolic_weight[n] = pow(fmax(true_weight[n], 1.0), 0.75);
  }

  // --- total TDN consumed ---
  vector[N] Ntotal;
  for (n in 1:N) {
    Ntotal[n] = rate_silage * obs_tdn_silage[n]
              + rate_tahu   * obs_tdn_tahu[n]
              + rate_SP2A   * obs_tdn_SP2A[n]
              + rate_SMG    * obs_tdn_SMG[n]
              + rate_rumput * obs_tdn_rumput[n];
  }

  // --- maintenance energy ---
  vector[N] Nmaintenance;
  for (n in 1:N) {
    real modifier = beta_BEF         * hasBEF_dt[n]
                  + beta_hormone     * hormone_effect_dt[n]
                  + beta_dewormed    * gotDewormed_dt[n]
                  + beta_vaccination * gotHNMVaccination_dt[n];
    Nmaintenance[n] = 0.077 * metabolic_weight[n] * exp(modifier);
  }

  // net energy for gain — clamped to prevent -inf propagation
  vector[N] NEg;
  for (n in 1:N) {
    NEg[n] = fmax(fmin(Ntotal[n] - Nmaintenance[n], 50.0), -10.0);
  }

  // --- energy-balance predicted ADG per animal ---
  // averaged across all observations for that animal — once per animal
  vector[n_animals] adg_energy;
  adg_energy = rep_vector(0.0, n_animals);
  {
    vector[n_animals] counts = rep_vector(0.0, n_animals);
    for (n in 1:N) {
      int a = animal_id[n];
      adg_energy[a] += beta_NEg * NEg[n];
      counts[a]     += 1.0;
    }
    for (a in 1:n_animals) {
      adg_energy[a] /= counts[a];
    }
    for (a in 1:n_animals) {
      // divide AND clamp in one step — counts only exists inside this block
      adg_energy[a] = fmax(fmin(adg_energy[a] / counts[a], 5.0), -2.0);
    }
  }
}

model {
  // --- weight priors ---
  weight_raw   ~ std_normal();         // non-centered innovations
  mu_init      ~ normal(350, 80);
  sigma_init   ~ exponential(0.033);
  sigma_obs    ~ exponential(0.12);
  sigma_process ~ exponential(0.5);

  // --- growth rate priors ---
  mu_growth    ~ normal(1.0, 0.3);
  sigma_growth ~ exponential(3);
  growth_rate_z ~ std_normal();

  // --- soft mechanistic constraint: once per animal ---
  // growth_rate should be consistent with energy balance prediction
  // sigma=0.3 means we allow ~0.3 kg/day deviation from energy prediction
  growth_rate ~ normal(adg_energy, 0.3);

  // --- consumption rates ---
  rate_silage ~ beta(9, 1);
  rate_tahu   ~ beta(9, 1);
  rate_SP2A   ~ beta(9, 1);
  rate_SMG    ~ beta(9, 1);
  rate_rumput ~ beta(9, 1);

  // --- maintenance modifiers ---
  beta_BEF         ~ normal(0,    0.2);
  beta_hormone     ~ normal(-0.1, 0.1);
  beta_dewormed    ~ normal(-0.1, 0.1);
  beta_vaccination ~ normal(0.1,  0.1);

  // --- energy conversion ---
  beta_NEg ~ normal(0.2, 0.05);

  // --- observation model ---
  obs_weight ~ normal(true_weight, sigma_obs);
}

generated quantities {
  vector[N] pred_weight;
  vector[N] residual;
  vector[N] adg;

  for (n in 1:N) {
    pred_weight[n] = normal_rng(true_weight[n], sigma_obs);
    residual[n]    = obs_weight[n] - true_weight[n];

    if (is_first[n] == 1) {
      adg[n] = 0;
    } else if (is_last[n] == 1) {
      adg[n] = not_a_number();
    } else {
      adg[n] = (true_weight[next_idx[n]] - true_weight[n]) / next_gap[n];
    }
  }
}
