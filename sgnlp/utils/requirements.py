import logging

from pkg_resources import parse_requirements, get_distribution, DistributionNotFound


def check_requirements(requirements, additional_message=None):
    missing_requirements = []
    mismatched_requirements = []
    parsed_requirements = parse_requirements(requirements)
    for requirement in parsed_requirements:
        try:
            installed_version = get_distribution(requirement.project_name).version
            assert installed_version in requirement
        except DistributionNotFound:
            missing_requirements.append(str(requirement))
        except AssertionError:
            mismatched_requirements.append((installed_version, requirement))

    if len(mismatched_requirements) > 0:
        error_msg = ""
        for mismatched_requirement in mismatched_requirements:
            error_msg += (
                f"\t"
                f"Package: {mismatched_requirement[1].project_name}, "
                f"Installed version: {mismatched_requirement[0]}, "
                f"Recommended version(s): {mismatched_requirement[1]}\n"
            )
        logging.warning(
            f"The following dependencies are mismatched and the model might not work as expected: "
            f"\n{error_msg}"
        )

    if len(missing_requirements) > 0:
        error_msg = f"To use this model, please install the following dependencies: {', '.join(missing_requirements)}."
        if additional_message:
            error_msg += "\n" + additional_message

        raise ModuleNotFoundError(error_msg)
