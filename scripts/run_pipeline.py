"""THE DRIVER: orchestrates stages 0..17 per PLAN.md §10/§15."""

import hydra
from omegaconf import DictConfig

from regworld.logging_conf import setup_logging
from regworld.pipeline import run_pipeline
from regworld.seeding import seed_everything
from regworld.tracking import make_tracker
from regworld.types import validate_config


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
