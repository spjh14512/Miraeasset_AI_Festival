"""Builders for embedding-ready contextual text."""

from vector_db.contextual_text_builders.kv_table_context_builder import (
    build_kv_contextual_text,
)
from vector_db.contextual_text_builders.r_table_record_context_builder import (
    build_r_table_contextual_text,
    build_r_table_record_contextual_text,
)
from vector_db.contextual_text_builders.r_table_row_group_builder import (
    RTableRowGroup,
    RowGroupRecordTooLargeError,
    build_r_table_row_groups,
)
from vector_db.contextual_text_builders.r_table_descriptor_builder import (
    build_r_table_descriptor_contextual_text,
)
from vector_db.contextual_text_builders.text_context_builder import (
    build_text_contextual_text,
)

__all__ = [
    "build_kv_contextual_text",
    "build_r_table_descriptor_contextual_text",
    "build_r_table_row_groups",
    "build_r_table_contextual_text",
    "build_r_table_record_contextual_text",
    "build_text_contextual_text",
    "RTableRowGroup",
    "RowGroupRecordTooLargeError",
]
