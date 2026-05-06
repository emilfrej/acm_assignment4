
// Kalman Filter Prototype Model — Single Subject, with Process Noise
// Key design:
//   log_r and log_q are the two parameters (both unconstrained).

//   r_value = exp(log_r): observation noise variance, how much variability is in a stimulus given its true category, 
//   i.e. how "spread out" each category is in feature space

//   q_value = exp(log_q): process noise / prototype drift rate, how fast the prototype drifts over time,
//    i.e. how much the agent expects categories to shift trial-to-trial

//   Kalman loop structure: Predict (add Q) → Decide → Update.
//   The full filter is in transformed parameters so log_lik in generated
//   quantities reads p[i] directly without re-running the filter.

data {
  int<lower=1> ntrials;
  int<lower=1> nfeatures; // I.e. eyes on stalks, spots, teeth etc (we have 5 in total)
  array[ntrials] int<lower=0, upper=1> cat_one; // true category of the alien on trial i (0 or 1)
  array[ntrials] int<lower=0, upper=1> y; // agent's response on trial i (0 or 1), this is what we are trying to predict
  array[ntrials, nfeatures] real obs; // A matrix of size ntrials x nfeatures (that meaning obs[trial x] = [1,0,0,1,0] fx)

  vector[nfeatures] initial_mu_cat0; // The start prototype for category 0, i.e. not dangerous, a vector of length 5 (nfeatures) equaling to one value per feature
  vector[nfeatures] initial_mu_cat1; // The start prototype for category 1, i.e. dangerous, same as above
  real<lower=0> initial_sigma_diag; // initial uncertainty about each prototype, this will go into a diagonal covariance matrix below. Large values = very uncertain

// Hyperparams for the two free parameters we are estimating described above 
  real prior_logr_mean;
  real<lower=0> prior_logr_sd;
  real prior_logq_mean;
  real<lower=0> prior_logq_sd;
}

// Unconstrained log-scale parameters, Stan samples these values freely
parameters {
  real log_r;
  real log_q; // A high q_value means the participant thinks categories are non-stationary and updates aggressively. 
              // A low q_value means they think categories are stable and update slowly.
}

// Its in the transformed parameters the kalman filter is applied 
transformed parameters {
  //start with transforming the unconstrained params into positive numbers only:
  real<lower=0> r_value = exp(log_r); 
  real<lower=0> q_value = exp(log_q);

  // Create an aray for storing the predicted repsonse probability p for each trial. 
  // The lower an upper bounds clamp it away from being exactly 0 or 1, as this would cause trouble for the bernoulli likelihood
  array[ntrials] real<lower=1e-9, upper=1-1e-9> p;

  // Local variables are declared below inside the {}:
  {
    // Begin with defining the same intital prototype means for each category as passed in the data chunk, these will be updated trial by trial:
    vector[nfeatures] mu_cat0 = initial_mu_cat0; 
    vector[nfeatures] mu_cat1 = initial_mu_cat1;
    
    // Create two prototype uncertainty covariance matrix, one for each category, using the initial uncertainty value passed in the data chunk
    // The diag_matrix() function creates a diagonal matrix (initial_sigma_diag on the diagonal and 0s everywhere else), which assumes indepence between features 
    matrix[nfeatures, nfeatures] sigma_cat0 =
      diag_matrix(rep_vector(initial_sigma_diag, nfeatures));
    matrix[nfeatures, nfeatures] sigma_cat1 =
      diag_matrix(rep_vector(initial_sigma_diag, nfeatures));
      
    // Create a diagonal matrix, this time with the r_value
    // This means all features have the same observation noise, and noise is independent across features.
    matrix[nfeatures, nfeatures] r_matrix =
      diag_matrix(rep_vector(r_value, nfeatures));
    // Create another diagonal matrix, this itme with the q_value, same implications as described above
    matrix[nfeatures, nfeatures] q_matrix =
      diag_matrix(rep_vector(q_value, nfeatures));
    // Lastly create a diagonal identity matrix, needed later in the covariance update formula
    matrix[nfeatures, nfeatures] I_mat =
      diag_matrix(rep_vector(1.0, nfeatures));

// Now we loop through each trial
    for (i in 1:ntrials) {
      vector[nfeatures] x = to_vector(obs[i]); //Convert the row in obs[i] into a Stan vector x, needed for matrix maths apparently 

      // ── Prediction step: add process noise to both categories ──────────
      sigma_cat0 = sigma_cat0 + q_matrix; // As described, this adds process noise to the uncertainty covariance matrix
      sigma_cat1 = sigma_cat1 + q_matrix; // This increases uncertainty before seeing the stimulus

      // ── Decision ────────────────────────────────────────────────────────
      // Add prototype uncertainty (sigma_cat) and observation noise (r_matrix) together to create a total covariance matrix for each category.
      matrix[nfeatures, nfeatures] cov0 = sigma_cat0 + r_matrix; 
      matrix[nfeatures, nfeatures] cov1 = sigma_cat1 + r_matrix;

      real log_p0 = multi_normal_lpdf(x | mu_cat0, cov0); // log-likelihood of the stimulus under each category 0's Gaussian distribution. Ie. how probable is the stimulus given category 0's prototype and spread?
      real log_p1 = multi_normal_lpdf(x | mu_cat1, cov1); // Same as above just for category 1
      
      // Now the two log-likelihoods are converted to a probability of category 1 via softmax over the two options: 
      real prob1  = exp(log_p1 - log_sum_exp(log_p0, log_p1));
      // This probability is stored in p[i] and its clamped away from being exactly 0 or 1, because bernoulli does not like that
      p[i] = fmax(1e-9, fmin(1 - 1e-9, prob1));

      // ── Update (measurement update for the correct category only) ────────
      if (cat_one[i] == 1) { //If the correct category was 1, then that prototype is updated:
        vector[nfeatures] innov = x - mu_cat1; //prediciton error (x-mu_prev in course notes)
        matrix[nfeatures, nfeatures] S  = sigma_cat1 + r_matrix; //combine prototype uncertainty (sigma_cat1) and observation noise (r_matrix) to create a total uncertainty matrix of the predicted observation
        matrix[nfeatures, nfeatures] K  = mdivide_right_spd(sigma_cat1, S); //Calculate the Kalmain Gain (solves for K = sigma_cat1 · S⁻¹), K~1 = trust the new observation a lot, update strongly, and K~0 = trust your prior more, update weakly
        matrix[nfeatures, nfeatures] IK = I_mat - K; //For simplifaction purposes, used in the Joseph form below where (I-K) is used multiple times
        mu_cat1    = mu_cat1 + K * innov; // Update the prototype mean, equalling to "updating your guess" from course notes
        sigma_cat1 = IK * sigma_cat1 * IK' + K * r_matrix * K'; //Update the prototype uncertainty using the Joseph form, called "Updating Your Confidence" in course notes
        sigma_cat1 = 0.5 * (sigma_cat1 + sigma_cat1'); //Force symmetry by averaging the matrix with its own transpose.
      
      } else { //If the correct category was 0, then that prototype is updated. All steps are equal to the above, just with cat0 instead:
        vector[nfeatures] innov = x - mu_cat0;
        matrix[nfeatures, nfeatures] S  = sigma_cat0 + r_matrix;
        matrix[nfeatures, nfeatures] K  = mdivide_right_spd(sigma_cat0, S);
        matrix[nfeatures, nfeatures] IK = I_mat - K;
        mu_cat0    = mu_cat0 + K * innov;
        sigma_cat0 = IK * sigma_cat0 * IK' + K * r_matrix * K';
        sigma_cat0 = 0.5 * (sigma_cat0 + sigma_cat0');
      }
    }
  } 
}

model {
  // Add log prior probabilities for log_r and log_q, these are gaussian distributions generated from the values passed in the data chunk
  target += normal_lpdf(log_r | prior_logr_mean, prior_logr_sd);
  target += normal_lpdf(log_q | prior_logq_mean, prior_logq_sd);
  
  // Binary response with success probability p, outcome is the decision on trial i: is the alien dangerous or not?
  target += bernoulli_lpmf(y | p); // Stan will push log_r and log_q toward values that make p[i] close to y[i] across trials
}

generated quantities {
  vector[ntrials] log_lik; //create a vector of per-trial log likelihoods
  real lprior; // create a scalar for the total log prior.
  
  // For leave-one-out cross validation (LOO):
  for (i in 1:ntrials)
    log_lik[i] = bernoulli_lpmf(y[i] | p[i]); // For each trial, compute the log probability of the agent's actual response given the model's predicted probability p[i].
  
  // For prior/posterior plots, saves the total log prior probability of the sampled parameter values
  lprior = normal_lpdf(log_r | prior_logr_mean, prior_logr_sd) +
           normal_lpdf(log_q | prior_logq_mean, prior_logq_sd);
}

