from dotenv import dotenv_values, load_dotenv


def load_env() -> dict:
    """Load environment variables from a .env file in the root of the project.

    Parameters
    ----------
    None

    Returns
    -------
    dict
        A dictionary of the environment variables that were loaded
        from the .env file. If the .env file was not found, or if
        there was an error loading the .env file, an empty dictionary
        is returned.
    """
    is_loaded = load_dotenv()

    if is_loaded:
        env_vars = dotenv_values()

        # Check that all keys are valid
        for key in env_vars:
            if not isinstance(key, str):
                raise ValueError(
                    f"Invalid environment variable key: {key}. Key must be a string."
                )

            if not key.replace("-", "_").isidentifier():
                raise ValueError(
                    f"Invalid environment variable key: {key} -- "
                    "contains invalid characters"
                )
            if not env_vars[key]:
                raise ValueError(
                    "Invalid environment variable value: "
                    f"{env_vars[key]} -- empty value"
                )
            if key.replace(" ", "") != key:
                raise ValueError(
                    "Invalid environment variable key: " f"{key} -- contains spaces"
                )

        return env_vars

    return {}
