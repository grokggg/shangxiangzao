"""Base model classes with enhanced robustness for tool inputs."""

import json

from pydantic import BaseModel, model_validator


class RobustBaseModel(BaseModel):
    """BaseModel that automatically converts str-annotated fields to strings.

    This handles edge cases where XML/JSON parsers may parse numeric strings
    as integers (e.g., "123" -> 123), which would fail pydantic validation.
    """

    @model_validator(mode='before')
    @classmethod
    def convert_str_fields(cls, data):
        """Convert all str-annotated fields to strings before validation."""
        if not isinstance(data, dict):
            return data
        for field_name, field_info in cls.model_fields.items():
            if field_info.annotation is str and field_name in data and data[field_name] is not None:
                if isinstance(data[field_name], str):
                    # Already a string, no need to convert
                    pass
                else:
                    # Convert to JSON string for non-string values to recover original value in tool_parser.py
                    data[field_name] = json.dumps(data[field_name], ensure_ascii=False)
        return data
