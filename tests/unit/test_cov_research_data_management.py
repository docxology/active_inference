"""Coverage tests for active_inference.research.data_management."""

import hashlib
import json
import re
from pathlib import Path

import pytest

from active_inference.research.data_management import (
    DataCollectionConfig,
    DataFormat,
    DataManager,
    DataSecurityLevel,
)


@pytest.fixture
def manager(tmp_path: Path) -> DataManager:
    return DataManager(tmp_path / "data", {"auto_validate": True})


def _collect(manager: DataManager, source: str = "simulation", **kwargs) -> str:
    kwargs.setdefault("collection_method", "sampling")
    return manager.collect_data(DataCollectionConfig(source_type=source, **kwargs))


class TestCollectData:
    @pytest.mark.parametrize(
        "source,expected",
        [
            ("simulation", {"simulation_data": "placeholder"}),
            ("experiment", {"experiment_data": "placeholder"}),
            ("api", {"api_data": "placeholder"}),
        ],
    )
    def test_sources_store_data_and_metadata(self, manager, source, expected):
        dataset_id = _collect(
            manager, source, preprocessing_steps=["normalize", "filter", "transform"]
        )

        stored = json.loads(
            (manager.base_dir / "processed_data" / f"{dataset_id}.json").read_text()
        )
        assert stored == expected

        metadata = manager.datasets[dataset_id]
        assert metadata.name == "sampling"
        assert metadata.description == "Data collected via sampling"
        assert metadata.format is DataFormat.JSON
        assert metadata.security_level is DataSecurityLevel.INTERNAL
        assert metadata.tags == ["normalize", "filter", "transform"]
        assert metadata.checksum == hashlib.md5(str(expected).encode()).hexdigest()

        saved_meta = json.loads(
            (manager.base_dir / "metadata" / f"{dataset_id}.json").read_text()
        )
        assert saved_meta["dataset_id"] == dataset_id
        assert saved_meta["format"] == "json"
        assert saved_meta["security_level"] == "internal"
        assert saved_meta["size_bytes"] == metadata.size_bytes
        assert saved_meta["checksum"] == metadata.checksum
        assert saved_meta["tags"] == ["normalize", "filter", "transform"]

    def test_unsupported_source_raises(self, manager):
        with pytest.raises(NotImplementedError, match="Unsupported source type: x"):
            _collect(manager, "x")
        assert manager.datasets == {}

    @pytest.mark.parametrize(
        "kwargs,field",
        [
            ({"source_type": "", "collection_method": "m"}, "source_type"),
            (
                {"source_type": "simulation", "collection_method": ""},
                "collection_method",
            ),
        ],
    )
    def test_missing_required_field(self, manager, kwargs, field):
        with pytest.raises(ValueError, match=f"Missing required field: {field}"):
            manager.collect_data(DataCollectionConfig(**kwargs))

    def test_validation_failure_raises_and_stores_nothing(self, manager, monkeypatch):
        monkeypatch.setattr(manager, "_collect_simulation_data", lambda _c: {})
        with pytest.raises(ValueError, match=re.escape("['Empty data']")):
            _collect(manager)
        assert list((manager.base_dir / "processed_data").iterdir()) == []
        assert manager.datasets == {}

    def test_unknown_preprocessing_step_is_ignored(self, manager):
        data = {"v": 1}
        assert manager._apply_preprocessing(data, ["unknown"]) is data


class TestValidation:
    def test_invalid_format(self, manager, monkeypatch):
        monkeypatch.setattr(manager, "_check_data_format", lambda _d, _f: False)
        result = manager._validate_data({"a": 1}, {"required_format": "csv"})
        assert result == {
            "valid": False,
            "issues": ["Invalid data format"],
            "warnings": [],
        }

    def test_quality_warnings(self, manager, monkeypatch):
        monkeypatch.setattr(manager, "_has_missing_values", lambda _d: True)
        monkeypatch.setattr(manager, "_has_outliers", lambda _d: True)
        result = manager._validate_data({"a": 1}, {})
        assert result["valid"] is True
        assert result["warnings"] == [
            "Data contains missing values",
            "Data contains potential outliers",
        ]

    def test_clean_data_has_no_quality_issues(self, manager):
        assert manager._check_data_quality({"a": 1}) == []
        assert manager._check_data_format({"a": 1}, "dict") is True


class TestStorage:
    def test_hdf5_route(self, manager, monkeypatch):
        calls = []
        monkeypatch.setattr(
            manager, "_determine_storage_format", lambda _d, _c: DataFormat.HDF5
        )
        monkeypatch.setattr(
            manager, "_store_hdf5_data", lambda i, d: calls.append(("hdf5", i, d))
        )
        manager._store_dataset("id1", {"a": 1}, DataCollectionConfig("s", "m"))
        assert calls == [("hdf5", "id1", {"a": 1})]
        assert not (manager.base_dir / "processed_data" / "id1.json").exists()

    def test_custom_route(self, manager, monkeypatch):
        calls = []
        monkeypatch.setattr(
            manager, "_determine_storage_format", lambda _d, _c: DataFormat.NPZ
        )
        monkeypatch.setattr(
            manager,
            "_store_custom_data",
            lambda i, d, f: calls.append(("custom", i, d, f)),
        )
        manager._store_dataset("id2", {"a": 1}, DataCollectionConfig("s", "m"))
        assert calls == [("custom", "id2", {"a": 1}, DataFormat.NPZ)]

    def test_placeholder_writers_create_no_files(self, manager):
        assert manager._store_hdf5_data("h", {"a": 1}) is None
        assert manager._store_custom_data("c", {"a": 1}, DataFormat.CSV) is None
        assert list((manager.base_dir / "processed_data").iterdir()) == []

    def test_data_size_matches_stored_file(self, manager):
        data = {"key": "value", "values": [1, 2, 3]}
        manager._store_json_data("sized", data)
        path = manager.base_dir / "processed_data" / "sized.json"
        assert manager._calculate_data_size(data) == path.stat().st_size


class TestRetrievalAndIntegrity:
    def test_get_unknown_dataset(self, manager):
        assert manager.get_dataset("missing") is None

    def test_get_dataset_roundtrip(self, manager):
        dataset_id = _collect(manager)
        assert manager.get_dataset(dataset_id) == {"simulation_data": "placeholder"}
        assert [m.dataset_id for m in manager.list_datasets()] == [dataset_id]

    def test_get_dataset_missing_file(self, manager):
        dataset_id = _collect(manager)
        (manager.base_dir / "processed_data" / f"{dataset_id}.json").unlink()
        assert manager.get_dataset(dataset_id) is None

    def test_get_dataset_corrupt_json(self, manager):
        dataset_id = _collect(manager)
        (manager.base_dir / "processed_data" / f"{dataset_id}.json").write_text("{")
        assert manager.get_dataset(dataset_id) is None

    def test_get_dataset_checksum_mismatch_still_returns_data(self, manager, caplog):
        dataset_id = _collect(manager)
        path = manager.base_dir / "processed_data" / f"{dataset_id}.json"
        path.write_text(json.dumps({"tampered": True}))
        with caplog.at_level("WARNING"):
            assert manager.get_dataset(dataset_id) == {"tampered": True}
        assert f"Checksum mismatch for dataset {dataset_id}" in caplog.text

    def test_integrity_of_untouched_dataset(self, manager):
        dataset_id = _collect(manager)
        assert manager.validate_dataset_integrity(dataset_id) == {
            "valid": True,
            "issues": [],
            "metadata": {
                "size_match": True,
                "checksum_match": True,
                "format_valid": True,
            },
        }

    def test_integrity_detects_tampering(self, manager):
        dataset_id = _collect(manager)
        path = manager.base_dir / "processed_data" / f"{dataset_id}.json"
        path.write_text(json.dumps({"simulation_data": "tampered!"}, indent=2))
        result = manager.validate_dataset_integrity(dataset_id)
        assert result["valid"] is False
        assert result["issues"] == ["File size mismatch", "Checksum mismatch"]
        assert result["metadata"]["size_match"] is False
        assert result["metadata"]["checksum_match"] is False

    def test_integrity_unknown_dataset(self, manager):
        assert manager.validate_dataset_integrity("nope") == {
            "valid": False,
            "error": "Dataset not found",
        }

    def test_integrity_unloadable_dataset(self, manager):
        dataset_id = _collect(manager)
        (manager.base_dir / "processed_data" / f"{dataset_id}.json").unlink()
        assert manager.validate_dataset_integrity(dataset_id) == {
            "valid": False,
            "error": "Could not load dataset",
        }


class TestBackup:
    def test_backup_writes_copy(self, manager):
        dataset_id = _collect(manager)
        backup = manager.backup_dataset(dataset_id)
        assert backup.parent == manager.base_dir / "backups"
        assert re.fullmatch(rf"{dataset_id}_\d{{8}}_\d{{6}}\.json", backup.name)
        assert json.loads(backup.read_text()) == {"simulation_data": "placeholder"}

    def test_backup_unknown_dataset(self, manager):
        assert manager.backup_dataset("missing") is None
        assert list((manager.base_dir / "backups").iterdir()) == []

    def test_backup_unloadable_dataset(self, manager):
        dataset_id = _collect(manager)
        (manager.base_dir / "processed_data" / f"{dataset_id}.json").unlink()
        assert manager.backup_dataset(dataset_id) is None
        assert list((manager.base_dir / "backups").iterdir()) == []
