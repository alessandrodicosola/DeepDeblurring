from pathlib import Path


def get_project_dir() -> Path:
    return Path(__file__).parent.parent


def get_data_dir() -> Path:
    return get_project_dir() / "data"


def get_cifar10_dir() -> Path:
    return get_data_dir() / "CIFAR10"


def get_reds_dir() -> Path:
    return get_data_dir() / "REDS"


def get_models_dir() -> Path:
    return get_project_dir() / "models"
