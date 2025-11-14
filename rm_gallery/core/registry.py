from typing import Callable, Dict, List, Type, Union

from loguru import logger

from rm_gallery.core.grader import Grader, GraderInfo, GraderMode
from rm_gallery.core.model.template import RequiredField


class GraderRegistry:
    """Registry for managing grader with hierarchical structure support."""

    _graders: Dict[str, Union[Grader | Callable, Dict]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        mode: GraderMode,
        description: str,
        required_fields: List[RequiredField],
        grader: Grader | Callable | Type[Grader] | None = None,
        namespace: str | None = None,
        **kwargs,
    ) -> Union[None, Callable, Grader]:
        """Register a grader function with a given name, optionally under a namespace.
        Can be used as a decorator, direct function call, or to initialize and register a grader.

        Args:
            name: The name of the grader
            mode: The grader mode
            description: The description of the grader
            required_fields: The required fields for the grader
            grader: The grader function to register (if used as direct function call)
                   Can be:
                   - An instance of Grader
                   - A callable function
                   - A Grader class to be instantiated
                   - None (for decorator usage)
            namespace: Optional namespace to group graders (e.g., "math", "code")
            **kwargs: Parameters for initializing the grader

        Returns:
            None if used as direct function call, decorator function if used as decorator,
            or the initialized grader if using initialization mode
        """
        # Create GraderInfo from individual parameters
        info = GraderInfo(
            name=name,
            mode=mode,
            description=description,
            required_fields=required_fields,
        )

        # If grader is provided, register it directly or initialize it if it's a class
        if grader is not None:
            # Check if grader is a class that needs to be instantiated
            if isinstance(grader, type) and issubclass(grader, Grader):
                # Instantiate the grader class with provided info and kwargs
                initialized_grader = grader(
                    name=info.name,
                    mode=info.mode,
                    description=info.description,
                    required_fields=info.required_fields,  # type: ignore
                    **kwargs,
                )
                cls._register_grader(info, initialized_grader, namespace)
            else:
                # Register the already instantiated grader or callable
                cls._register_grader(info, grader, namespace)
            return None

        # If grader is not provided, return a decorator
        def decorator(
            target: Type[Grader] | Grader | Callable,
        ) -> Type[Grader] | Grader | Callable:
            # Check if target is a class that needs to be instantiated
            if isinstance(target, type) and issubclass(target, Grader):
                # Instantiate the grader class with provided info and kwargs
                initialized_grader = target(
                    name=info.name,
                    mode=info.mode,
                    description=info.description,
                    required_fields=info.required_fields,  # type: ignore
                    **kwargs,
                )
                cls._register_grader(info, initialized_grader, namespace)
                return target
            else:
                # Register the already instantiated grader or callable
                cls._register_grader(info, target, namespace)
                return target

        return decorator

    @classmethod
    def _register_grader(
        cls,
        info: GraderInfo,
        grader: Grader | Callable,
        namespace: str | None = None,
    ) -> None:
        """Internal method to register a grader function.

        Args:
            info: The GraderInfo containing grader metadata
            grader: The grader function to register
            namespace: Optional namespace to group graders (e.g., "math", "code")
        """
        # If the grader is a callable but not a Grader instance, wrap it with FunctionGrader
        if not isinstance(grader, Grader):
            from rm_gallery.core.grader import FunctionGrader

            # Create a new list to ensure type compatibility
            required_fields = list(info.required_fields)
            wrapped_grader = FunctionGrader(
                func=grader,
                name=info.name,
                mode=info.mode,
                description=info.description,
                required_fields=required_fields,  # type: ignore
            )
            grader = wrapped_grader

        # Use name from GraderInfo
        name = info.name
        full_name = f"{namespace}.{name}" if namespace is not None else name

        # Update grader with info from GraderInfo if it's not already set
        if not grader.name:
            grader.name = full_name
        if grader.mode is None:
            grader.mode = info.mode
        if not grader.description:
            grader.description = info.description
        if not grader.required_fields:
            grader.required_fields = info.required_fields

        # Handle namespace creation
        if namespace is not None:
            if namespace not in cls._graders:
                cls._graders[namespace] = {}
            elif not isinstance(cls._graders[namespace], dict):
                raise ValueError(
                    f"Namespace '{namespace}' conflicts with an existing grader name"
                )

            namespace_dict = cls._graders[namespace]
            if not isinstance(namespace_dict, dict):
                raise ValueError(f"Namespace '{namespace}' is not a valid namespace")

            if name in namespace_dict:
                logger.warning(
                    f"grader '{full_name}' is already registered. Overwriting."
                )

            namespace_dict[name] = grader
        else:
            if name in cls._graders and isinstance(cls._graders[name], dict):
                raise ValueError(
                    f"grader name '{name}' conflicts with an existing namespace"
                )

            if name in cls._graders:
                logger.warning(f"grader '{name}' is already registered. Overwriting.")

            cls._graders[name] = grader

        logger.info(f"Registered grader '{full_name}'")

    @classmethod
    def get(cls, name: str) -> Grader | None:
        """Get a registered grader function by name (supports dot notation for namespaces).

        Args:
            name: The name of the grader function to get (e.g., "math.accuracy" or "general")

        Returns:
            The registered grader function, or None if not found
        """
        if "." in name:
            # Handle namespaced graders
            namespace, sub_name = name.split(".", 1)
            if namespace in cls._graders and isinstance(cls._graders[namespace], dict):
                namespace_dict = cls._graders[namespace]
                if isinstance(namespace_dict, dict) and sub_name in namespace_dict:
                    return namespace_dict[sub_name]
            return None
        else:
            # Handle direct graders
            grader = cls._graders.get(name)
            if isinstance(grader, Grader):
                return grader
            return None

    @classmethod
    def remove(cls, name: str) -> bool:
        """Remove a registered grader function by name (supports dot notation for namespaces).

        Args:
            name: The name of the grader function to remove (e.g., "math.accuracy" or "general")

        Returns:
            True if the grader was removed, False if it wasn't registered
        """
        if "." in name:
            # Handle namespaced graders
            namespace, sub_name = name.split(".", 1)
            if namespace in cls._graders and isinstance(cls._graders[namespace], dict):
                namespace_dict = cls._graders[namespace]
                if isinstance(namespace_dict, dict) and sub_name in namespace_dict:
                    del namespace_dict[sub_name]
                    # Clean up empty namespace
                    if not namespace_dict:
                        del cls._graders[namespace]
                    logger.info(f"Removed grader '{name}'")
                    return True
            return False
        else:
            # Handle direct graders
            if name in cls._graders and isinstance(cls._graders[name], Grader):
                del cls._graders[name]
                logger.info(f"Removed grader '{name}'")
                return True
            return False

    @classmethod
    def list_graders(cls, namespace: str | None = None) -> Dict[str, str]:
        """List registered graders, optionally filtered by namespace.

        Args:
            namespace: Optional namespace to filter by

        Returns:
            A dictionary mapping grader names to their types
        """
        result = {}

        if namespace:
            # List graders in a specific namespace
            if namespace in cls._graders and isinstance(cls._graders[namespace], dict):
                namespace_dict = cls._graders[namespace]
                if isinstance(namespace_dict, dict):
                    for name, grader in namespace_dict.items():
                        if isinstance(grader, Grader):
                            result[f"{namespace}.{name}"] = type(grader).__name__
            return result
        else:
            # List all graders
            for key, value in cls._graders.items():
                if isinstance(value, Grader):
                    # Direct grader
                    result[key] = type(value).__name__
                elif isinstance(value, dict):
                    # Namespace
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, Grader):
                            result[f"{key}.{sub_key}"] = type(sub_value).__name__
            return result

    @classmethod
    def list_namespaces(cls) -> list:
        """List all available namespaces.

        Returns:
            A list of namespace names
        """
        return [key for key, value in cls._graders.items() if isinstance(value, dict)]

    @classmethod
    def clear(cls) -> None:
        """Clear all registered graders."""
        cls._graders.clear()
        logger.info("Cleared all registered graders")


GR = GraderRegistry
