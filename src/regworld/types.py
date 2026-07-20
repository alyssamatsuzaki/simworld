"""Pydantic mirrors of every Hydra config group (§6). `validate_config` fails fast on typos.

Models use `extra="forbid"`: a misspelled key dies in the first second, not after six hours.
"""

from __future__ import annotations

from typing import Any, Literal

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PathsCfg(_Cfg):
    root: str = "artifacts"
    data: str = "artifacts/data"
    graphs: str = "artifacts/graphs"
    reports: str = "reports"


class StagesCfg(_Cfg):
    """Driver switchboard. Defaults False so `stages={}` runs nothing (§10 gate 0)."""

    recon: bool = False
    data: bool = False
    graphs: bool = False
    abm: bool = False
    tensorized_abm: bool = False
    calibration: bool = False
    causal: bool = False
    emulator: bool = False
    envs: bool = False
    marl: bool = False
    rl: bool = False
    ensemble: bool = False
    sensitivity: bool = False
    figures: bool = False
    report: bool = False


class ComputeCfg(_Cfg):
    name: Literal["local", "ray_local", "ray_cluster", "slurm"] = "ray_local"
    address: str | None = None  # ray cluster address; None -> local ray.init()
    num_cpus: int | None = None  # cap Ray CPUs; None -> all


class DataCfg(_Cfg):
    source: Literal["synthetic", "real"] = "synthetic"
    real_panel_path: str | None = None  # the swap point (§1); used only when source="real"


class DgpCfg(_Cfg):
    variant: Literal["wellspecified", "confounded", "misspecified"] = "confounded"
    corr_z_size: float = 0.35  # corr(latent capacity, log size); 0 under wellspecified
    decision_rule: Literal["logit", "logit_interacted"] = "logit"  # misspecified DGP adds a term
    sigma_obs: float = 0.01  # aggregate observation noise
    panel_sample_frac: float = 0.20  # firm panel sampling (§7.9)
    survey_sample_frac: float = 0.40  # consumer survey segment sampling
    misclassification: float = 0.05  # reported-compliance flip rate (q0=q1 at truth)
    edge_dropout: float = 0.20  # observed graph: share of true edges missing
    edge_spurious: float = 0.03  # observed graph: spurious edge share


class PopulationCfg(_Cfg):
    name: Literal["small", "base", "large"] = "base"
    n_firms: int = 2000
    n_consumer_segments: int = 20
    n_regions: int = 8
    n_sectors: int = 6
    n_associations: int = 4


class NetworkCfg(_Cfg):
    name: Literal["scalefree", "smallworld", "empirical"] = "scalefree"
    supply_m: int = 2  # preferential-attachment edges per new node
    alpha: float = 1.0  # degree exponent in P(i->j)
    homophily: float = 1.5  # lambda_homoph; the confounding knob (§7.2)
    ws_k: int = 6  # Watts-Strogatz neighbours (segment influence)
    ws_p: float = 0.1  # Watts-Strogatz rewiring
    firms_per_segment: int = 8  # market edges per segment


class BehaviorCfg(_Cfg):
    name: Literal["logit_baseline", "logit_sticky", "bounded_rational"] = "logit_sticky"
    sticky: bool = True  # switching-cost term active in the fitted/simulated rule
    attention: float = 1.0  # bounded_rational: probability a firm reconsiders each quarter


class AbmCfg(_Cfg):
    vectorized: bool = True  # NumPy across the AgentSet for the firm decision (§16 g14)
    collect_agent_panel: bool = True
    max_quarters: int = 24


class ObjectiveCfg(_Cfg):
    name: Literal["balanced", "compliance_first", "competition_first"] = "balanced"
    w_c: float = 1.0
    w_h: float = 0.5
    w_s: float = 0.5
    w_e: float = 0.1
    w_t: float = 0.3
    w_x: float = 0.3


class NutsCfg(_Cfg):
    warmup: int = 1000
    draws: int = 1000
    chains: int = 4


class SmcAbcCfg(_Cfg):
    particles: int = 2000
    rounds: int = 4  # SMC populations
    quantile: float = 0.5  # per-round epsilon quantile


class CalibrationCfg(_Cfg):
    method: Literal["numpyro_nuts", "smc_abc", "numpyro_bsl", "pymc_nuts"] = "numpyro_nuts"
    crosscheck: bool = True  # PyMC re-implementation of the micro model (4c)
    design_points: int = 256
    replicates: int = 8
    nuts: NutsCfg = Field(default_factory=NutsCfg)
    smc_abc: SmcAbcCfg = Field(default_factory=SmcAbcCfg)
    device: Literal["cpu", "gpu"] = "cpu"
    did_penalty: float = 0.0  # >0 after a FLAGGED 5f gate: moment-match the DiD estimate


class CausalCfg(_Cfg):
    e_low: float = 0.2
    e_high: float = 0.8
    n_do_seeds: int = 64  # DGP do() replications for tau_true
    on_disagreement: Literal["recalibrate", "report"] = "recalibrate"
    run_discovery: bool = True  # 5e PC/GES vs true DAG
    refuter_subset_frac: float = 0.8


class EmulatorCfg(_Cfg):
    arch: Literal["rssm_gnn", "rssm_flat", "gru_baseline"] = "rssm_gnn"
    stochastic_level: Literal["macro", "node"] = "macro"
    latent_categories: int = 32  # discrete latents: variables
    latent_classes: int = 32  # classes per variable
    deter_dim: int = 256
    hidden_dim: int = 128
    gnn_layers: int = 3
    train_episodes: int = 2000
    epochs: int = 40
    train_steps: int = 30000
    batch_size: int = 16
    seq_len: int = 24
    burn_in: int = 8
    imag_horizon: int = 8  # open-loop imagination loss horizon (k)
    lr: float = 3.0e-4
    kl_free: float = 1.0  # free bits (nats)
    kl_balance: float = 0.8  # KL balancing toward the prior
    grad_clip: float = 100.0
    compile: bool = False
    reward_from_outcomes: bool = True  # recompute reward from decoded outcomes (§10 St.8)


class EnvCfg(_Cfg):
    name: Literal["single_agent", "multi_agent"] = "single_agent"
    graph_obs: bool = False
    n_strategic_firms: int = 10  # PettingZoo: top-K learners


class PolicyCfg(_Cfg):
    name: str = "phased_targeted"
    kind: Literal["static", "learned"] = "static"
    enforcement: float = 0.6
    targeting: float = 0.5
    phase_speed: float = 0.4
    subsidy: float = 0.3
    source: Literal["sb3", "dreamer", "bo", "none"] = "none"  # kind="learned": which artifact


class RlCfg(_Cfg):
    algo: Literal["sb3_ppo", "sb3_sac", "torchrl_dreamer", "rllib_marl"] = "sb3_ppo"
    total_timesteps: int = 300000
    n_envs: int = 8
    train_dreamer: bool = True
    marl_timesteps: int = 200000
    marl_backend: Literal["rllib", "ippo"] = "ippo"  # RLlib is non-gating (§16 g11)


class EnsembleCfg(_Cfg):
    name: Literal["grid_small", "grid_full"] = "grid_small"
    posterior_draws: int = 1000
    n_seeds: int = 3
    policies: list[str] = Field(
        default_factory=lambda: [
            "none",
            "uniform_low",
            "uniform_high",
            "targeted",
            "phased_targeted",
            "rl_ppo",
            "rl_dreamer",
        ]
    )
    validation_frac: float = 0.05  # ABM cross-validation subsample (§10 St.11)
    batch_size: int = 64  # rollouts per Ray task


class SensitivityCfg(_Cfg):
    method: Literal["morris", "sobol"] = "sobol"
    morris_trajectories: int = 8
    sobol_n: int = 1024  # Saltelli N
    top_k: int = 8  # parameters kept after the Morris screen
    optuna_trials: int = 12
    bo_evals: int = 40
    abm_check_points: int = 64  # emulator-vs-ABM agreement subsample


class TrackingCfg(_Cfg):
    backend: Literal["mlflow", "wandb", "none"] = "mlflow"
    uri: str = "sqlite:///experiments/mlflow.db"  # file store is EOL in mlflow 3.x
    experiment: str = "regworld"


class EvalCfg(_Cfg):
    name: Literal["fast", "full"] = "fast"
    abm_validation_episodes: int = 64
    k_steps: list[int] = Field(default_factory=lambda: [1, 3, 6, 12, 18, 24])
    n_dist_rollouts: int = 256  # distributional-fidelity ensemble size per side


class RegWorldConfig(_Cfg):
    profile_name: Literal["smoke", "dev", "full"] = "smoke"
    force_stage: str | None = None  # re-run this stage and everything downstream (§15)
    isolated_envs: bool = False  # run each stage as a subprocess (§5 fallback)
    seed: int = 0
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])
    horizon_quarters: int = 24
    observed_quarters: int = 12
    device: Literal["auto", "cpu", "cuda"] = "auto"
    paths: PathsCfg = Field(default_factory=PathsCfg)
    stages: StagesCfg = Field(default_factory=StagesCfg)
    compute: ComputeCfg = Field(default_factory=ComputeCfg)
    data: DataCfg = Field(default_factory=DataCfg)
    dgp: DgpCfg = Field(default_factory=DgpCfg)
    population: PopulationCfg = Field(default_factory=PopulationCfg)
    network: NetworkCfg = Field(default_factory=NetworkCfg)
    behavior: BehaviorCfg = Field(default_factory=BehaviorCfg)
    abm: AbmCfg = Field(default_factory=AbmCfg)
    objective: ObjectiveCfg = Field(default_factory=ObjectiveCfg)
    calibration: CalibrationCfg = Field(default_factory=CalibrationCfg)
    causal: CausalCfg = Field(default_factory=CausalCfg)
    emulator: EmulatorCfg = Field(default_factory=EmulatorCfg)
    env: EnvCfg = Field(default_factory=EnvCfg)
    policy: PolicyCfg = Field(default_factory=PolicyCfg)
    rl: RlCfg = Field(default_factory=RlCfg)
    ensemble: EnsembleCfg = Field(default_factory=EnsembleCfg)
    sensitivity: SensitivityCfg = Field(default_factory=SensitivityCfg)
    tracking: TrackingCfg = Field(default_factory=TrackingCfg)
    eval: EvalCfg = Field(default_factory=EvalCfg)

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:  # pragma: no cover
            return "cpu"


def validate_config(cfg: DictConfig | dict[str, Any]) -> RegWorldConfig:
    """OmegaConf -> resolved dict -> validated Pydantic model. Raises on any unknown key."""
    container = OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else cfg
    if not isinstance(container, dict):  # pragma: no cover - defensive
        raise TypeError(f"config root must be a mapping, got {type(container)}")
    mapping: dict[str, Any] = {str(k): v for k, v in container.items() if k != "hydra"}
    return RegWorldConfig(**mapping)
