data {
  int<lower=1> N;
  int<lower=1> n_animals;

  array[N] int<lower=1>         animal_id;
  vector[N]                     obs_weight;
  vector[N]                     days_gap;
  array[N] int<lower=0,upper=1> is_first;
  array[N] int<lower=1>         prev_idx;

  // fixed from Kalman smoother — not estimated
  real<lower=0> sigma_obs;
  real<lower=0> sigma_process;
}

parameters {
  // per-animal growth rates (partial pooling)
  real<lower=0>              mu_growth;
  real<lower=0>              sigma_growth;
  vector<lower=0>[n_animals] growth_rate;

  // initial weight
  real          mu_init;
  real<lower=0> sigma_init;

  // latent true weight
  vector[N] true_weight;
}

model {
  // --- priors ---
  mu_growth    ~ normal(1.2, 0.4);
  sigma_growth ~ normal(0.3, 0.1) T[0, ];

  mu_init    ~ normal(350, 50);
  sigma_init ~ normal(50, 15) T[0, ];

  growth_rate ~ normal(mu_growth, sigma_growth);

  // --- state transitions ---
  for (n in 1:N) {
    int a = animal_id[n];
    if (is_first[n] == 1) {
      true_weight[n] ~ normal(mu_init, sigma_init);
    } else {
      true_weight[n] ~ normal(
        true_weight[prev_idx[n]] + growth_rate[a] * days_gap[n],
        sigma_process * sqrt(days_gap[n])
      );
    }
  }

  // --- observation model ---
  obs_weight ~ normal(true_weight, sigma_obs);
}

generated quantities {
  vector[N] pred_weight;
  vector[N] residual;

  for (n in 1:N) {
    pred_weight[n] = normal_rng(true_weight[n], sigma_obs);
    residual[n]    = obs_weight[n] - true_weight[n];
  }
}
