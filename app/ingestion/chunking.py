from dataclasses import asdict, dataclass
import hashlib
import re

@dataclass
class ParentChunk:
    parent_id: str
    document_id: str
    source: str
    text: str
    start_char: int
    end_char: int

@dataclass
class ChildChunk:
    child_id: str
    parent_id: str
    document_id: str
    source: str
    text: str
    start_char: int
    end_char: int

def _id(*parts: str) -> str:
    return hashlib.sha1("::".join(parts).encode("utf-8")).hexdigest()

def _spans(text: str, size: int, overlap: int) -> list[tuple[int, int]]:
    if size <= 0:
        raise ValueError("Chunk size must be positive.")
    if overlap < 0 or overlap >= size:
        raise ValueError("Overlap must be >= 0 and smaller than chunk size.")
    spans, start, length = [], 0, len(text)
    while start < length:
        hard_end = min(start + size, length)
        end = hard_end
        if hard_end < length:
            floor = start + int(size * 0.65)
            boundary = max(text.rfind(" ", floor, hard_end), text.rfind("\n", floor, hard_end))
            if boundary > start:
                end = boundary
        if end <= start:
            end = hard_end
        spans.append((start, end))
        if end >= length:
            break
        start = max(start + 1, end - overlap)
        while start < length and text[start].isspace():
            start += 1
    return spans

def create_parent_child_chunks(text: str, source: str, parent_size: int = 1000,
                               parent_overlap: int = 100, child_size: int = 200,
                               child_overlap: int = 30):
    normalized = re.sub(r"\r\n?", "\n", text).strip()
    if not normalized:
        return [], []
    document_id = _id(source, normalized)
    parents, children = [], []
    for pi, (ps, pe) in enumerate(_spans(normalized, parent_size, parent_overlap)):
        ptext = normalized[ps:pe].strip()
        if not ptext:
            continue
        pid = _id(document_id, "parent", str(pi), ptext)
        parents.append(ParentChunk(pid, document_id, source, ptext, ps, pe))
        for ci, (cs, ce) in enumerate(_spans(ptext, child_size, child_overlap)):
            ctext = ptext[cs:ce].strip()
            if ctext:
                children.append(ChildChunk(_id(pid, "child", str(ci), ctext), pid,
                                           document_id, source, ctext, ps + cs, ps + ce))
    return parents, children

def chunks_to_dicts(chunks):
    return [asdict(chunk) for chunk in chunks]
