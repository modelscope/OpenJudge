# -*- coding: utf-8 -*-
from typing import Any, Callable, Dict, List, Union

from jsonschema import ValidationError, validate
from pydantic import BaseModel, Field

from rm_gallery.core.utils import get_value_by_mapping, get_value_by_path


class DataSample(BaseModel):
    """Data sample containing shared data and individual samples.

    DataSample is the basic data structure for evaluation tasks. It consists of
    shared context data and independent samples to be evaluated.

    For pointwise evaluation: Each DataSample contains one sample in the samples list.
    For listwise evaluation: Each DataSample contains multiple samples in the samples list.

    Attributes:
        data (dict): A dictionary containing shared data for all samples
                    (e.g., query, reference answer).
        samples (List[dict]): A list of dictionaries, each representing an
                             individual sample to evaluate.
    """

    data: dict = Field(
        default_factory=dict, description="Shared data for all samples"
    )
    samples: List[dict] = Field(
        default_factory=list,
        description="List of individual samples to evaluate",
    )


# DataSampleParser is a type alias for either a dictionary mapping or a callable function
# that takes a DataSample and returns a DataSample
DataSampleParser = Union[
    Dict[str, str], Callable[["DataSample"], "DataSample"]
]


def parse_data_sample(
    data_sample: DataSample,
    parser: DataSampleParser | None,
) -> DataSample:
    """Parse a data sample using a parser.

    Args:
        data_sample (DataSample): A data sample to parse.
        parser (DataSampleParser | None): A parser to use for parsing the data sample.
            Can be either:
            1. A dictionary with direct field mappings where paths start with "data" or "sample"
            2. A callable function that takes a DataSample and returns a DataSample
            3. None, in which case the data sample is returned unchanged

    Returns:
        DataSample: The parsed data sample.

    Raises:
        ValueError: If parser is invalid type.
    """
    if parser is None:
        return data_sample

    # Handle dictionary configuration
    if isinstance(parser, dict):
        # Apply mappings to the main data dictionary
        data = data_sample.data.copy()
        samples = [sample.copy() for sample in data_sample.samples]

        # Process mappings based on path prefix
        for target_field, source_path in parser.items():
            path_parts = source_path.split(".")
            if not path_parts:
                continue

            # Check the first part of the path to determine data source
            if path_parts[0] == "data":
                # Get value from data
                if len(path_parts) > 1:
                    value = get_value_by_path(
                        data_sample.data, ".".join(path_parts[1:])
                    )
                else:
                    value = data_sample.data
                data[target_field] = value

            elif path_parts[0] == "sample":
                # Apply to each sample in samples list
                for i, sample in enumerate(data_sample.samples):
                    if len(path_parts) > 1:
                        value = get_value_by_path(
                            sample, ".".join(path_parts[1:])
                        )
                    else:
                        value = sample
                    if len(samples) <= i:
                        samples.append({})
                    samples[i][target_field] = value
            else:
                raise ValueError(f"Invalid path prefix: {path_parts[0]}")

        return DataSample(data=data, samples=samples)

    # Handle callable directly
    if callable(parser):
        return parser(data_sample)

    raise ValueError(f"Invalid parser type: {type(parser)}")


def validate_data_samples(
    data_samples: List[dict | DataSample],
    schema: dict | None = None,
):
    # Validate that data_sample_schema is provided
    if not schema:
        raise ValueError("the schema of data sample is required")

    # Validate that all data samples conform to data_sample_schema
    for i, sample in enumerate(data_samples):
        try:
            # If it's already a DataSample, convert to dict for validation
            if isinstance(sample, DataSample):
                # For DataSample objects, we validate the 'data' part against the schema
                sample_dict = sample.data
            else:
                # For dict objects, validate directly
                sample_dict = sample

            validate(instance=sample_dict, schema=schema)
        except ValidationError as e:
            raise ValueError(
                f"Data sample at index {i} does not conform to schema: {str(e)}",
            )
        except Exception as e:
            raise ValueError(
                f"Error validating data sample at index {i}: {str(e)}"
            )

    return [
        DataSample(**data_sample)
        if isinstance(data_sample, dict)
        else data_sample
        for data_sample in data_samples
    ]
