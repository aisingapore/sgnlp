import importlib


def check_requirements(requirements):
    missing_requirements = []
    for requirement in requirements:
        try:
            importlib.import_module(requirement)
        except ModuleNotFoundError:
            missing_requirements.append(requirement)
    if len(missing_requirements) > 0:
        raise ModuleNotFoundError(
            f"To use this model, please install the following dependencies: {', '.join(missing_requirements)}")
