# -*- coding: utf-8 -*-
"""
Data Mapping Utilities for OpenJudge

This module provides utility functions for extracting and mapping data from
nested dictionaries and lists using path-based queries. It is primarily used
in OpenJudge's evaluation system to map task and workflow output data to
evaluator parameters.

Key Features:
- Path-based data extraction from nested structures
- Automatic list traversal for batch processing
- Flexible data mapping with dictionary or callable mappers

The primary use case is mapping task data structures to evaluator input parameters,
especially in listwise evaluation scenarios where multiple workflow outputs need
to be processed together.
"""

from typing import Any, Callable, Dict, List, Optional, Union


# pylint: disable=too-many-return-statements
def get_value_by_path(
    data: Union[Dict[str, Any], List[Any]],
    path: str,
) -> Optional[Any]:
    """Get value from dictionary by path, supporting list indexing and automatic list traversal.

    This function retrieves a value from a nested data structure (dictionary or list)
    using a dot-separated path. It supports both dictionary key access and list indexing.

    Special feature: When accessing a list with a field path like "items.field", it will
    automatically traverse the list and extract the specified field from each item,
    returning a list of those values.

    Args:
        data: The data dictionary or list to query.
        path: Dot-separated path, e.g. "item.ticket_text" or "items.0.name" or "items.name".

    Returns:
        Any: The value at the path, or None if path doesn't exist.

    Example:
        >>> data = {"user": {"name": "Alice", "emails": ["alice@example.com", "alice@work.com"]}}
        >>> get_value_by_path(data, "user.name")
        'Alice'
        >>> get_value_by_path(data, "user.emails.0")
        'alice@example.com'
        >>> get_value_by_path(data, "user.phone") is None
        True
        >>> data = {"items": [{"name": "A"}, {"name": "B"}]}
        >>> get_value_by_path(data, "items.name")
        ['A', 'B']
    """
    if not path:
        return data

    keys = path.split(".")
    key = keys[0]
    remaining_path = ".".join(keys[1:]) if len(keys) > 1 else ""

    # Handle list data
    if isinstance(data, list):
        # If key is a digit, treat as index
        if key.isdigit():
            try:
                index = int(key)
                if index < len(data):
                    return get_value_by_path(data[index], remaining_path)
                else:
                    return None
            except (ValueError, IndexError):
                return None
        else:
            # Auto-traverse list and extract field from each item
            result = []
            for item in data:
                item_value = get_value_by_path(item, path)
                if item_value is not None:
                    result.append(item_value)
            return result if result else None

    # Handle dict data
    elif isinstance(data, dict):
        if key in data:
            return get_value_by_path(data[key], remaining_path)
        else:
            return None

    # Handle other data types
    else:
        # Can't traverse further
        if not key:  # If no more keys, return the data
            return data
        else:
            # Trying to access a key on a non-dict/non-list
            return None


def get_value_by_mapping(
    data: Dict[str, Any],
    mapping: Dict[str, str],
) -> Dict[str, Any]:
    """Get values from dictionary according to mapping.

    This function applies a mapping to transform a data dictionary, extracting
    values from specified paths and placing them under new field names.

    Args:
        data: The data dictionary to query.
        mapping: Mapping relationship, key is the new field name, value is the path to extract from.

    Returns:
        dict: Mapped dictionary with extracted values.

    Example:
        >>> data = {"user": {"name": "Alice", "age": 30}}
        >>> mapping = {"username": "user.name", "user_age": "user.age"}
        >>> result = get_value_by_mapping(data, mapping)
        >>> print(result)
        {'username': 'Alice', 'user_age': 30}
    """
    result: Dict[str, Any] = {}
    for field, path in mapping.items():
        result[field] = get_value_by_path(data, path)
    return result


def parse_data_with_mapper(
    data: dict,
    mapper: Dict[str, str] | Callable | None = None,
) -> dict:
    """Parse data with mapper.

    This function transforms data using either a dictionary-based mapping or a
    callable function. If a dictionary is provided, it uses get_value_by_mapping.
    If a callable is provided, it applies the function directly to the data.

    Args:
        data: The data dictionary to parse.
        mapper: Mapping relationship (key is new field name, value is path) or a callable function.

    Returns:
        dict: Parsed data.

    Example:
        >>> data = {"user": {"name": "Alice"}}
        >>> # Using dictionary mapper
        >>> mapper = {"username": "user.name"}
        >>> result = parse_data_with_mapper(data, mapper)
        >>> print(result)
        {'username': 'Alice'}
        >>>
        >>> # Using callable mapper
        >>> def uppercase_names(d):
        ...     return {k: v.upper() if isinstance(v, str) else v for k, v in d.items()}
        >>> result = parse_data_with_mapper({"name": "alice"}, uppercase_names)
        >>> print(result)
        {'name': 'ALICE'}
    """
    if isinstance(mapper, dict):
        data = get_value_by_mapping(data, mapper)
    elif callable(mapper):
        data = mapper(data)
    return data
