"""
Vbot WandB Experiment Tracking Configuration
=============================================
This module provides integration helpers for logging training experiments,
hyperparameters, and model checkpoints to Weights & Biases (WandB).
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("vbot.wandb")

# Default configurations
DEFAULT_PROJECT = "vbot-styletts2"
DEFAULT_ENTITY = None  # Uses default user/team account


class WandBLogger:
    """Handles initialization, logging, and artifact management for WandB."""

    def __init__(
        self,
        project_name: str = DEFAULT_PROJECT,
        entity: Optional[str] = DEFAULT_ENTITY,
        config_dict: Optional[Dict[str, Any]] = None,
        job_type: str = "train",
        tags: Optional[list] = None,
    ):
        self.project_name = project_name
        self.entity = entity
        self.config_dict = config_dict or {}
        self.job_type = job_type
        self.tags = tags or []
        self.run = None

        # Disable interactive prompt to avoid blocking automated pipelines
        os.environ["WANDB_SILENT"] = "true"

    def init_run(self, run_name: Optional[str] = None) -> Optional[Any]:
        """Initialize a new WandB run. Returns the run object if successful."""
        try:
            import wandb

            # Check if API key is present (or already configured in env)
            if not os.environ.get("WANDB_API_KEY") and not Path("~/.config/wandb/api.key").expanduser().exists():
                logger.warning("WandB API key not found. Logging will be offline.")
                os.environ["WANDB_MODE"] = "offline"

            self.run = wandb.init(
                project=self.project_name,
                entity=self.entity,
                config=self.config_dict,
                name=run_name,
                job_type=self.job_type,
                tags=self.tags,
                reinit=True,
            )
            logger.info(f"WandB initialized. Run: {self.run.name} (Mode: {wandb.run.mode})")
            return self.run
        except ImportError:
            logger.error("wandb package not installed. Run 'pip install wandb' to log experiments.")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize WandB run: {e}")
            return None

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log training or validation metrics for a specific training step."""
        if not self.run:
            return
        try:
            import wandb

            wandb.log(metrics, step=step)
        except Exception as e:
            logger.error(f"Error logging metrics to WandB: {e}")

    def log_model_artifact(
        self,
        checkpoint_path: str,
        artifact_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        aliases: Optional[list] = None,
    ):
        """Upload and version a model checkpoint (e.g. .pt, .pth) as a WandB artifact."""
        if not self.run:
            logger.warning("No active WandB run. Artifact upload skipped.")
            return
        try:
            import wandb

            file_path = Path(checkpoint_path)
            if not file_path.exists():
                logger.error(f"Checkpoint file not found: {checkpoint_path}")
                return

            aliases = aliases or ["latest"]
            metadata = metadata or {}

            # Create a model artifact
            artifact = wandb.Artifact(
                name=artifact_name, type="model", description=f"Model checkpoint at step/epoch", metadata=metadata
            )

            # Add file to artifact
            if file_path.is_dir():
                artifact.add_dir(str(file_path))
            else:
                artifact.add_file(str(file_path))

            # Log the artifact
            self.run.log_artifact(artifact, aliases=aliases)
            logger.info(f"Successfully uploaded model artifact '{artifact_name}' to WandB.")
        except Exception as e:
            logger.error(f"Error saving artifact to WandB: {e}")

    def finish_run(self):
        """Finish the active WandB run cleanly."""
        if self.run:
            try:
                import wandb

                wandb.finish()
                logger.info("WandB run closed successfully.")
            except Exception as e:
                logger.error(f"Error closing WandB run: {e}")
            self.run = None
