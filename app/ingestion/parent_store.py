import json
from pathlib import Path

class ParentDocumentStore:
    """Persistent JSON repository for larger parent passages."""
    def __init__(self, storage_path="data/parent_documents.json"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._parents = {}
        if self.storage_path.exists():
            data = json.loads(self.storage_path.read_text(encoding="utf-8"))
            self._parents = {item["parent_id"]: item for item in data}

    def add_many(self, parents):
        for parent in parents:
            self._parents[parent["parent_id"]] = parent
        self._save()

    def get(self, parent_id):
        return self._parents.get(parent_id)

    def get_many(self, parent_ids):
        result, seen = [], set()
        for pid in parent_ids:
            if pid not in seen and pid in self._parents:
                result.append(self._parents[pid])
                seen.add(pid)
        return result

    def all(self):
        return list(self._parents.values())

    def _save(self):
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.storage_path.with_suffix(self.storage_path.suffix + ".tmp")
        temp.write_text(json.dumps(self.all(), ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self.storage_path)
