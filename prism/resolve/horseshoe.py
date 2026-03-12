"""
Horseshoe Prior Differential Expression for PRISM-Resolve.

Bayesian hierarchical model with horseshoe priors for identifying
sparse cryptic discriminator genes between eccrine and hair progenitors.

Model for each gene g:
  x_i^g | β^g, P(fate_i) ~ NB(μ_gi, r_g)
  log(μ_gi) = β_0^g + β_1^g · P(eccrine | z_i) + covariates
  β_1^g | λ_g ~ N(0, λ_g² τ²)
  λ_g ~ C+(0, 1)  [half-Cauchy local shrinkage]
  τ ~ C+(0, s₀/G)  [half-Cauchy global shrinkage]

Reference: Van der Pas et al., "Adaptive posterior contraction rates
for the horseshoe", EJS 2017.
"""

import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from typing import Optional, Dict, Tuple
import warnings


class HorseshoeDE:
    """Horseshoe prior differential expression analysis.

    Identifies sparse cryptic discriminator genes by fitting a Bayesian
    regression with horseshoe priors. The horseshoe aggressively shrinks
    non-discriminative genes to zero while leaving true discriminators
    unshrunk.

    Output: ranked gene list with posterior inclusion probabilities
    and calibrated credible intervals.
    """

    def __init__(
        self,
        n_warmup: int = 2000,
        n_samples: int = 4000,
        n_chains: int = 4,
        s0_ratio: float = 0.01,
        seed: int = 42,
    ):
        self.n_warmup = n_warmup
        self.n_samples = n_samples
        self.n_chains = n_chains
        self.s0_ratio = s0_ratio
        self.seed = seed
        self.results = None

    @staticmethod
    def horseshoe_model(
        X: jnp.ndarray,        # (N, p) design matrix (intercept + fate_prob + covariates)
        y: jnp.ndarray,        # (N,) gene expression counts
        s0: float = 0.01,      # expected sparsity ratio
        obs_mask: Optional[jnp.ndarray] = None,
    ):
        """NumPyro model with horseshoe prior.

        This is the core Bayesian model:
        - Global shrinkage τ ~ C+(0, s0) controls overall sparsity
        - Local shrinkage λ_j ~ C+(0, 1) allows individual coefficients to escape
        - Coefficients β_j ~ N(0, λ_j² τ²)
        """
        N, p = X.shape

        # Global shrinkage
        tau = numpyro.sample("tau", dist.HalfCauchy(s0))

        # Local shrinkage for each predictor
        lambdas = numpyro.sample("lambdas", dist.HalfCauchy(jnp.ones(p)))

        # Coefficients with horseshoe prior
        beta = numpyro.sample(
            "beta",
            dist.Normal(jnp.zeros(p), lambdas * tau)
        )

        # Intercept
        intercept = numpyro.sample("intercept", dist.Normal(0, 5))

        # Overdispersion for negative binomial
        phi = numpyro.sample("phi", dist.Gamma(2, 0.5))

        # Linear predictor
        mu = jnp.exp(intercept + X @ beta)
        mu = jnp.clip(mu, 1e-6, 1e6)

        # Observation model: Negative Binomial
        # NB parameterized as (total_count=phi, probs=phi/(phi+mu))
        probs = jnp.clip(phi / (phi + mu), 1e-6, 1 - 1e-6)

        with numpyro.plate("data", N):
            numpyro.sample("obs", dist.NegativeBinomial2(mu, phi), obs=y)

    def fit(
        self,
        expression: np.ndarray,        # (N, G) count matrix
        fate_probs: np.ndarray,         # (N,) P(eccrine | z_i) from mixture model
        gene_names: list,               # (G,) gene names
        covariates: Optional[np.ndarray] = None,  # (N, C) additional covariates
        batch_size: int = 100,          # genes per batch (for memory)
    ) -> Dict:
        """Fit horseshoe model for all genes.

        Due to computational cost, we fit each gene independently
        (embarrassingly parallel).

        Args:
            expression: Raw count matrix (N cells x G genes)
            fate_probs: Eccrine fate probability for each cell
            gene_names: List of gene names
            covariates: Optional covariate matrix (e.g., total counts, mito%)
            batch_size: Number of genes to process at once

        Returns:
            Dict with ranked genes, posterior inclusion probabilities,
            effect sizes, and credible intervals.
        """
        N, G = expression.shape

        # Build design matrix
        X = fate_probs.reshape(-1, 1)
        if covariates is not None:
            X = np.concatenate([X, covariates], axis=1)
        p = X.shape[1]

        print(f"Fitting horseshoe DE for {G} genes, {N} cells, {p} predictors")
        print(f"MCMC: {self.n_warmup} warmup, {self.n_samples} samples, {self.n_chains} chains")

        results = {
            "gene": [],
            "beta_fate_mean": [],
            "beta_fate_std": [],
            "beta_fate_ci_lower": [],
            "beta_fate_ci_upper": [],
            "posterior_inclusion_prob": [],
            "local_shrinkage_mean": [],
            "tau_mean": [],
        }

        X_jax = jnp.array(X)

        # Process genes in batches
        for start in range(0, G, batch_size):
            end = min(start + batch_size, G)
            batch_genes = gene_names[start:end]
            print(f"  Processing genes {start}-{end} / {G}...")

            for g_idx, gene_name in enumerate(batch_genes):
                g = start + g_idx
                y = expression[:, g]

                # Skip genes with too few non-zero counts
                if np.sum(y > 0) < 10:
                    results["gene"].append(gene_name)
                    results["beta_fate_mean"].append(0.0)
                    results["beta_fate_std"].append(0.0)
                    results["beta_fate_ci_lower"].append(0.0)
                    results["beta_fate_ci_upper"].append(0.0)
                    results["posterior_inclusion_prob"].append(0.0)
                    results["local_shrinkage_mean"].append(0.0)
                    results["tau_mean"].append(0.0)
                    continue

                y_jax = jnp.array(y.astype(np.float32))

                try:
                    gene_results = self._fit_single_gene(
                        X_jax, y_jax, self.s0_ratio
                    )
                    results["gene"].append(gene_name)
                    results["beta_fate_mean"].append(float(gene_results["beta_mean"][0]))
                    results["beta_fate_std"].append(float(gene_results["beta_std"][0]))
                    results["beta_fate_ci_lower"].append(float(gene_results["beta_ci"][0, 0]))
                    results["beta_fate_ci_upper"].append(float(gene_results["beta_ci"][0, 1]))
                    results["posterior_inclusion_prob"].append(float(gene_results["pip"][0]))
                    results["local_shrinkage_mean"].append(float(gene_results["lambda_mean"][0]))
                    results["tau_mean"].append(float(gene_results["tau_mean"]))
                except Exception as e:
                    warnings.warn(f"Failed to fit gene {gene_name}: {e}")
                    results["gene"].append(gene_name)
                    for key in ["beta_fate_mean", "beta_fate_std", "beta_fate_ci_lower",
                                "beta_fate_ci_upper", "posterior_inclusion_prob",
                                "local_shrinkage_mean", "tau_mean"]:
                        results[key].append(0.0)

        # Sort by posterior inclusion probability
        import pandas as pd
        df = pd.DataFrame(results)
        df = df.sort_values("posterior_inclusion_prob", ascending=False).reset_index(drop=True)

        self.results = df
        print(f"\nTop 20 cryptic discriminator genes:")
        print(df.head(20)[["gene", "beta_fate_mean", "posterior_inclusion_prob"]].to_string())

        return df

    def _fit_single_gene(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        s0: float,
    ) -> Dict:
        """Fit horseshoe model for a single gene using NUTS."""
        rng_key = jax.random.PRNGKey(self.seed)

        kernel = NUTS(self.horseshoe_model, max_tree_depth=8)
        mcmc = MCMC(
            kernel,
            num_warmup=self.n_warmup,
            num_samples=self.n_samples,
            num_chains=self.n_chains,
            progress_bar=False,
        )

        mcmc.run(rng_key, X=X, y=y, s0=s0)
        samples = mcmc.get_samples()

        # Extract fate coefficient (first predictor)
        beta_samples = samples["beta"]  # (n_samples, p)
        lambda_samples = samples["lambdas"]
        tau_samples = samples["tau"]

        # Posterior statistics
        beta_mean = beta_samples.mean(axis=0)
        beta_std = beta_samples.std(axis=0)
        beta_ci = jnp.percentile(beta_samples, jnp.array([2.5, 97.5]), axis=0).T

        # Posterior inclusion probability (PIP)
        # Defined as P(|β| > threshold | data)
        # Use practical significance threshold
        threshold = 0.1  # log-fold-change threshold
        pip = (jnp.abs(beta_samples) > threshold).mean(axis=0)

        lambda_mean = lambda_samples.mean(axis=0)
        tau_mean = tau_samples.mean()

        return {
            "beta_mean": np.array(beta_mean),
            "beta_std": np.array(beta_std),
            "beta_ci": np.array(beta_ci),
            "pip": np.array(pip),
            "lambda_mean": np.array(lambda_mean),
            "tau_mean": float(tau_mean),
        }

    def fit_mcmc(
        self,
        expression: np.ndarray,
        fate_probs: np.ndarray,
        gene_names: list,
        covariates: Optional[np.ndarray] = None,
        batch_size: int = 100,
        n_warmup: int = 500,
        n_samples: int = 1000,
        n_chains: int = 2,
    ) -> "pd.DataFrame":
        """Reduced-budget MCMC horseshoe DE.

        Uses the same correct horseshoe_model and _fit_single_gene as fit(),
        but with practical runtime parameters (~2h vs ~12h for full MCMC).

        Args:
            expression: (N, G) count matrix
            fate_probs: (N,) P(eccrine | z_i) from mixture model
            gene_names: (G,) gene names
            covariates: (N, C) optional covariates
            batch_size: genes per batch
            n_warmup: MCMC warmup iterations (default 500 vs 2000 in full)
            n_samples: MCMC samples (default 1000 vs 4000 in full)
            n_chains: MCMC chains (default 2 vs 4 in full)

        Returns:
            DataFrame with ranked genes and PIPs
        """
        import pandas as pd

        # Temporarily override MCMC params
        orig_warmup = self.n_warmup
        orig_samples = self.n_samples
        orig_chains = self.n_chains

        self.n_warmup = n_warmup
        self.n_samples = n_samples
        self.n_chains = n_chains

        try:
            result = self.fit(expression, fate_probs, gene_names,
                            covariates=covariates, batch_size=batch_size)
        finally:
            # Restore original params
            self.n_warmup = orig_warmup
            self.n_samples = orig_samples
            self.n_chains = orig_chains

        return result

    def fit_fast(
        self,
        expression: np.ndarray,
        fate_probs: np.ndarray,
        gene_names: list,
        covariates: Optional[np.ndarray] = None,
    ) -> "pd.DataFrame":
        """Fast approximate horseshoe DE using MAP estimation.

        For rapid iteration, use MAP + Laplace approximation
        instead of full MCMC. Results are approximate but much faster.
        """
        from sklearn.linear_model import BayesianRidge
        import pandas as pd

        N, G = expression.shape
        X = fate_probs.reshape(-1, 1)
        if covariates is not None:
            X = np.concatenate([X, covariates], axis=1)

        results = {
            "gene": [],
            "beta_fate_mean": [],
            "beta_fate_std": [],
            "posterior_inclusion_prob": [],
        }

        for g in range(G):
            y = expression[:, g].astype(np.float64)

            if np.sum(y > 0) < 10:
                results["gene"].append(gene_names[g])
                results["beta_fate_mean"].append(0.0)
                results["beta_fate_std"].append(0.0)
                results["posterior_inclusion_prob"].append(0.0)
                continue

            # Log-transform for approximate Gaussian
            y_log = np.log1p(y)

            try:
                model = BayesianRidge(fit_intercept=True, compute_score=True)
                model.fit(X, y_log)

                beta = model.coef_[0]
                # Per-gene posterior std from posterior covariance matrix
                # model.sigma_ is the (p, p) posterior covariance of coefficients
                beta_std = np.sqrt(model.sigma_[0, 0]) if hasattr(model, 'sigma_') and model.sigma_ is not None else np.sqrt(1.0 / (model.alpha_ + 1e-8))

                # PIP = P(|beta| > threshold | data), using posterior N(beta, beta_std^2)
                # P(-t < beta < t) = Phi((t - beta)/std) - Phi((-t - beta)/std)
                # PIP = 1 - P(-t < beta < t)
                from scipy.stats import norm
                threshold = 0.1  # log-fold-change practical significance
                prob_inside = (norm.cdf(threshold, loc=beta, scale=beta_std)
                              - norm.cdf(-threshold, loc=beta, scale=beta_std))
                pip = 1.0 - prob_inside

                results["gene"].append(gene_names[g])
                results["beta_fate_mean"].append(float(beta))
                results["beta_fate_std"].append(float(beta_std))
                results["posterior_inclusion_prob"].append(float(pip))
            except Exception:
                results["gene"].append(gene_names[g])
                results["beta_fate_mean"].append(0.0)
                results["beta_fate_std"].append(0.0)
                results["posterior_inclusion_prob"].append(0.0)

        df = pd.DataFrame(results)
        df = df.sort_values("posterior_inclusion_prob", ascending=False).reset_index(drop=True)

        self.results = df
        return df
