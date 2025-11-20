# -*- coding: utf-8 -*-
"""instance."""
import importlib
from typing import Any, Type, TypedDict


class InstDict(TypedDict, total=False):
    """instance param."""

    class_name: str
    """Name of the class to instantiate."""
    module_path: str
    """Module path where the class is defined."""
    kwargs: dict
    """Keyword arguments to pass to the class constructor."""


Inst = InstDict | object


def init_instance_by_config(
    config: Inst,
    accept_type: Type | None = None,
) -> Any:
    """Initialize an instance from configuration dictionary or check existing instance.

    The configuration can be:
    - A dict containing 'class', 'module', and 'kwargs' for instantiation
    - An existing object instance that needs type checking

    Args:
        config (Inst): Configuration dictionary with class, module and kwargs,
                                           or an existing object instance
        accept_type (Type, optional): Expected type or base class that the instantiated
                                    class should be subclass of. If provided, will check
                                    if the instantiated class is a subclass of accept_type.

    Returns:
        Any: Initialized instance of the specified class or the existing instance

    Raises:
        TypeError: If accept_type is provided and the instantiated class is not
                  a subclass of accept_type, or if config is neither a dict nor valid instance

    Example:
        >>> # From config dict
        >>> config = {
        ...     'class': 'StringMatchGrader',
        ...     'module': 'rm_gallery.gallery.grader.text.string_match',
        ...     'kwargs': {'ignore_case': True}
        ... }
        >>> instance = init_instance_by_config(config)
        >>>
        >>> # With existing instance
        >>> existing_instance = StringMatchGrader(ignore_case=True)
        >>> instance = init_instance_by_config(existing_instance)
        >>>
        >>> # With type checking
        >>> from rm_gallery.core.grader.base import Grader
        >>> instance = init_instance_by_config(config, accept_type=Grader)
    """
    # If config is already an instance, just check its type
    if not isinstance(config, dict):
        instance = config
        if accept_type is not None and not isinstance(instance, accept_type):
            raise TypeError(
                f"Provided instance {instance.__class__.__name__} "
                f"is not an instance of {accept_type.__name__}",
            )
        return instance

    # Otherwise, instantiate from config dict
    class_name = config["class_name"]
    module_path = config["module_path"]
    kwargs = config.get("kwargs", {})

    # Import the module
    module = importlib.import_module(module_path)

    # Get the class from the module
    cls = getattr(module, class_name)

    # Check type if accept_type is provided
    if accept_type is not None and not issubclass(cls, accept_type):
        raise TypeError(
            f"Instantiated class {cls.__name__} is not a of {accept_type.__name__}",
        )

    # Instantiate the class with kwargs
    return cls(**kwargs)
