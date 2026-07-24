"""THE DRIVER: orchestrates stages 0..17 per PLAN.md §10/§15."""

import hydra
from omegaconf import DictConfig

from simworld.logging_conf import setup_logging
from simworld.pipeline import run_pipeline
from simworld.seeding import seed_everything
from simworld.tracking import make_tracker
from simworld.types import validate_config


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg_obj = validate_config(cfg)  # Pydantic — fails in the first second on a typo'd key
    seed_everything(cfg_obj.seed)
    setup_logging()
    tracker = make_tracker(cfg_obj)
    tracker.start("pipeline", cfg_obj.model_dump())
    try:
        manifest = run_pipeline(cfg_obj, tracker)
        stages = manifest["stages"]
        assert isinstance(stages, dict)
        for name, res in stages.items():
            tracker.log_metrics({f"stage_{name}_wall_clock": res["wall_clock"]})
    finally:
        tracker.finish()


if __name__ == "__main__":
    main()
