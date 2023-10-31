import mlflow
from omegaconf import DictConfig


def log_params_from_omegaconf_dict(config: DictConfig):
    """Log all the parameters in the config object.

    Args:
        config (DictConfig): The config object to log.
    """
    for key, value in config.items():
        if isinstance(value, DictConfig):
            log_params_from_omegaconf_dict(value)
        else:
            mlflow.log_param(str(key), value)
