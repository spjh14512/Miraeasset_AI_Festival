"""Resolve canonical source references against one loaded document."""

from __future__ import annotations

from dataclasses import dataclass
import xml.etree.ElementTree as ET

from converters.common.document_loader import LoadedDocument
from converters.common.source_models import SourceRef


def _tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1].upper()


@dataclass(frozen=True, slots=True)
class ResolvedElement:
    element: ET.Element
    element_path: str
    table_index: int | None


class SourceResolver:
    """Build O(1) lookup indexes using the same paths as the chunker."""

    def __init__(self, document: LoadedDocument) -> None:
        if document.root is None:
            raise ValueError("Cannot index a document without a root element")
        self.document = document
        self._by_path: dict[str, ResolvedElement] = {}
        self._by_html_id: dict[str, ResolvedElement] = {}
        self._by_element_id: dict[int, ResolvedElement] = {}
        self._tables: list[ResolvedElement] = []
        self._index(document.root, f"/{_tag(document.root)}[1]")

    def _index(self, element: ET.Element, path: str) -> None:
        table_index: int | None = None
        if _tag(element) == "TABLE":
            table_index = len(self._tables)
        resolved = ResolvedElement(element, path, table_index)
        self._by_path[path] = resolved
        self._by_element_id[id(element)] = resolved
        html_id = element.attrib.get("ID") or element.attrib.get("id")
        if html_id:
            self._by_html_id.setdefault(html_id, resolved)
        if table_index is not None:
            self._tables.append(resolved)

        counts: dict[str, int] = {}
        for child in list(element):
            child_tag = _tag(child)
            counts[child_tag] = counts.get(child_tag, 0) + 1
            self._index(child, f"{path}/{child_tag}[{counts[child_tag]}]")

    def resolve(self, source_ref: SourceRef) -> ResolvedElement:
        if source_ref.element_path:
            resolved = self._by_path.get(source_ref.element_path)
            if resolved is not None:
                return resolved
        if source_ref.html_id:
            resolved = self._by_html_id.get(source_ref.html_id)
            if resolved is not None:
                return resolved
        if source_ref.table_index is not None:
            try:
                return self._tables[source_ref.table_index]
            except IndexError:
                pass
        raise KeyError(f"Source reference could not be resolved: {source_ref.to_dict()}")

    def table_index(self, element: ET.Element) -> int | None:
        resolved = self._by_element_id.get(id(element))
        return resolved.table_index if resolved is not None else None

    def path(self, element: ET.Element) -> str:
        resolved = self._by_element_id.get(id(element))
        if resolved is None:
            raise KeyError("Element does not belong to this source document")
        return resolved.element_path


__all__ = ["ResolvedElement", "SourceResolver"]
