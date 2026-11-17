"""Behavioural tests for active_inference.tools.utilities."""

import json
import math
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from active_inference.tools.utilities import (
    ConfigurationManager,
    DataProcessingTools,
    FileManager,
    HelperFunctions,
    MathematicalUtilities,
    Utilities,
)


@pytest.fixture
def dp() -> DataProcessingTools:
    return DataProcessingTools()


class TestDataProcessingTools:
    def test_normalize_empty_returns_empty(self, dp: DataProcessingTools) -> None:
        assert dp.normalize_data([]) == []

    def test_normalize_minmax(self, dp: DataProcessingTools) -> None:
        assert dp.normalize_data([2.0, 4.0, 6.0]) == pytest.approx([0.0, 0.5, 1.0])

    def test_normalize_minmax_constant(self, dp: DataProcessingTools) -> None:
        assert dp.normalize_data([3.0, 3.0, 3.0]) == [0.0, 0.0, 0.0]

    def test_normalize_zscore(self, dp: DataProcessingTools) -> None:
        result = dp.normalize_data([1.0, 2.0, 3.0], method="zscore")
        std = math.sqrt(2.0 / 3.0)
        assert result == pytest.approx([-1 / std, 0.0, 1 / std])
        assert sum(result) == pytest.approx(0.0)

    def test_normalize_zscore_constant(self, dp: DataProcessingTools) -> None:
        assert dp.normalize_data([5.0, 5.0], method="zscore") == [0.0, 0.0]

    def test_normalize_unknown_method_returns_input(
        self, dp: DataProcessingTools
    ) -> None:
        data = [1.0, 9.0]
        assert dp.normalize_data(data, method="bogus") is data

    def test_smooth_short_data_unchanged(self, dp: DataProcessingTools) -> None:
        data = [1.0, 2.0]
        assert dp.smooth_data(data, window_size=5) is data

    def test_smooth_moving_average(self, dp: DataProcessingTools) -> None:
        result = dp.smooth_data([0.0, 3.0, 6.0, 9.0], window_size=3)
        # windows: [0,3], [0,3,6], [3,6,9], [6,9]
        assert result == pytest.approx([1.5, 3.0, 6.0, 7.5])

    def test_statistics_empty(self, dp: DataProcessingTools) -> None:
        assert dp.compute_statistics([]) == {}

    def test_statistics_values(self, dp: DataProcessingTools) -> None:
        stats = dp.compute_statistics([1.0, 2.0, 3.0, 10.0])
        assert stats["mean"] == pytest.approx(4.0)
        assert stats["median"] == pytest.approx(2.5)
        assert stats["min"] == 1.0
        assert stats["max"] == 10.0
        assert stats["count"] == 4
        assert stats["std"] == pytest.approx(math.sqrt(12.5))

    def test_resample_same_length_returns_input(self, dp: DataProcessingTools) -> None:
        data = [1.0, 2.0]
        assert dp.resample_data(data, 2) is data

    def test_resample_empty_gives_zeros(self, dp: DataProcessingTools) -> None:
        assert dp.resample_data([], 3) == [0.0, 0.0, 0.0]

    def test_resample_interpolates(self, dp: DataProcessingTools) -> None:
        assert dp.resample_data([0.0, 10.0], 5) == pytest.approx(
            [0.0, 2.5, 5.0, 7.5, 10.0]
        )


class TestConfigurationManager:
    def test_load_missing_file(self, tmp_path: Path) -> None:
        cm = ConfigurationManager()
        assert cm.load_config(tmp_path / "missing.json") == {}
        assert cm.config_cache == {}

    def test_load_json_and_cache(self, tmp_path: Path) -> None:
        path = tmp_path / "c.json"
        path.write_text(json.dumps({"a": 1}))
        cm = ConfigurationManager()
        assert cm.load_config(path) == {"a": 1}
        # Changing the file does not affect the cached value.
        path.write_text(json.dumps({"a": 2}))
        assert cm.load_config(path) == {"a": 1}
        assert cm.config_cache[path] == {"a": 1}

    @pytest.mark.parametrize("suffix", [".yml", ".yaml"])
    def test_load_yaml(self, tmp_path: Path, suffix: str) -> None:
        path = tmp_path / f"c{suffix}"
        path.write_text("a: 1\nb:\n  c: two\n")
        assert ConfigurationManager().load_config(path) == {"a": 1, "b": {"c": "two"}}

    def test_load_unknown_suffix_auto(self, tmp_path: Path) -> None:
        path = tmp_path / "c.ini"
        path.write_text("[x]")
        assert ConfigurationManager().load_config(path) == {}

    def test_load_explicit_json_type_overrides_suffix(self, tmp_path: Path) -> None:
        path = tmp_path / "c.txt"
        path.write_text(json.dumps({"k": [1, 2]}))
        assert ConfigurationManager().load_config(path, config_type="json") == {
            "k": [1, 2]
        }

    def test_load_explicit_unsupported_type(self, tmp_path: Path) -> None:
        path = tmp_path / "c.txt"
        path.write_text("anything")
        cm = ConfigurationManager()
        assert cm.load_config(path, config_type="toml") == {}
        assert cm.config_cache[path] == {}

    def test_load_malformed_json(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{not json")
        cm = ConfigurationManager()
        assert cm.load_config(path) == {}
        assert path not in cm.config_cache

    def test_save_json(self, tmp_path: Path) -> None:
        path = tmp_path / "out.json"
        cm = ConfigurationManager()
        assert cm.save_config({"x": 1}, path) is True
        assert json.loads(path.read_text()) == {"x": 1}
        assert cm.config_cache[path] == {"x": 1}

    def test_save_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "out.yaml"
        assert ConfigurationManager().save_config({"y": [1, 2]}, path, "yaml")
        assert yaml.safe_load(path.read_text()) == {"y": [1, 2]}

    def test_save_unsupported_format(self, tmp_path: Path) -> None:
        path = tmp_path / "out.toml"
        cm = ConfigurationManager()
        assert cm.save_config({"x": 1}, path, "toml") is False
        assert path not in cm.config_cache

    def test_save_failure(self, tmp_path: Path) -> None:
        # A directory cannot be opened for writing.
        assert ConfigurationManager().save_config({"x": 1}, tmp_path) is False

    def test_merge_configs_recursive(self) -> None:
        cm = ConfigurationManager()
        base = {"a": 1, "nested": {"x": 1, "y": 2}, "replace": {"k": 1}}
        override = {"b": 2, "nested": {"y": 3, "z": 4}, "replace": 5}
        merged = cm.merge_configs(base, override)
        assert merged == {
            "a": 1,
            "b": 2,
            "nested": {"x": 1, "y": 3, "z": 4},
            "replace": 5,
        }
        # Base config is not mutated at the top level.
        assert base["replace"] == {"k": 1}
        assert "b" not in base


class TestFileManager:
    def test_ensure_directory(self, tmp_path: Path) -> None:
        target = tmp_path / "a" / "b"
        assert FileManager().ensure_directory(target) is True
        assert target.is_dir()

    def test_ensure_directory_failure(self, tmp_path: Path) -> None:
        blocker = tmp_path / "file"
        blocker.write_text("x")
        assert FileManager().ensure_directory(blocker / "sub") is False

    def test_safe_write_file(self, tmp_path: Path) -> None:
        target = tmp_path / "deep" / "out.txt"
        assert FileManager().safe_write_file(target, "hello") is True
        assert target.read_text() == "hello"
        assert not (tmp_path / "deep" / "out.txt.tmp").exists()

    def test_safe_write_file_failure(self, tmp_path: Path) -> None:
        blocker = tmp_path / "file"
        blocker.write_text("x")
        assert FileManager().safe_write_file(blocker / "out.txt", "data") is False

    def test_json_roundtrip(self, tmp_path: Path) -> None:
        fm = FileManager()
        target = tmp_path / "nested" / "d.json"
        assert fm.save_json_file({"a": [1, 2]}, target) is True
        assert fm.load_json_file(target) == {"a": [1, 2]}

    def test_load_json_missing(self, tmp_path: Path) -> None:
        assert FileManager().load_json_file(tmp_path / "nope.json") is None

    def test_save_json_failure(self, tmp_path: Path) -> None:
        blocker = tmp_path / "file"
        blocker.write_text("x")
        assert FileManager().save_json_file({"a": 1}, blocker / "d.json") is False


class TestMathematicalUtilities:
    def test_softmax_default(self) -> None:
        mu = MathematicalUtilities()
        result = mu.softmax([0.0, math.log(3.0)])
        assert result == pytest.approx([0.25, 0.75])

    def test_softmax_temperature(self) -> None:
        mu = MathematicalUtilities()
        result = mu.softmax([0.0, 2 * math.log(3.0)], temperature=2.0)
        assert result == pytest.approx([0.25, 0.75])

    def test_softmax_numerically_stable(self) -> None:
        result = MathematicalUtilities().softmax([1000.0, 1000.0])
        assert result == pytest.approx([0.5, 0.5])

    def test_normalize_probabilities_shifts_negatives(self) -> None:
        result = MathematicalUtilities().normalize_probabilities([-1.0, 1.0])
        assert sum(result) == pytest.approx(1.0)
        assert all(p > 0 for p in result)
        assert result[1] == pytest.approx(2.000001 / 2.000002)

    def test_entropy(self) -> None:
        mu = MathematicalUtilities()
        assert mu.compute_entropy([0.5, 0.5, 0.0]) == pytest.approx(math.log(2))
        assert mu.compute_entropy([0.0, 0.0]) == 0.0

    def test_kl_divergence(self) -> None:
        mu = MathematicalUtilities()
        assert mu.kl_divergence([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)
        expected = 0.5 * math.log(0.5 / 0.25) + 0.5 * math.log(0.5 / 0.75)
        assert mu.kl_divergence([0.5, 0.5], [0.25, 0.75]) == pytest.approx(expected)


class TestHelperFunctions:
    def test_format_timestamp_explicit(self) -> None:
        ts = datetime(2024, 1, 2, 3, 4, 5)
        assert HelperFunctions().format_timestamp(ts) == "2024-01-02T03:04:05"

    def test_format_timestamp_now(self) -> None:
        before = datetime.now()
        parsed = datetime.fromisoformat(HelperFunctions().format_timestamp())
        assert before <= parsed <= datetime.now()

    def test_generate_id(self) -> None:
        hf = HelperFunctions()
        first = hf.generate_id("run", length=6)
        second = hf.generate_id("run", length=6)
        assert first.startswith("run_")
        assert len(first) == len("run_") + 6
        assert first != second
        assert len(hf.generate_id()) == len("id_") + 8

    def test_flatten_dict(self) -> None:
        nested = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
        assert HelperFunctions().flatten_dict(nested) == {
            "a": 1,
            "b.c": 2,
            "b.d.e": 3,
        }

    def test_deep_merge(self) -> None:
        merged = HelperFunctions().deep_merge(
            {"a": {"x": 1, "y": 1}, "b": 1}, {"a": {"y": 2}, "b": {"z": 1}}
        )
        assert merged == {"a": {"x": 1, "y": 2}, "b": {"z": 1}}


class TestUtilities:
    def test_components_wired(self) -> None:
        u = Utilities({"k": "v"})
        assert u.config == {"k": "v"}
        assert isinstance(u.data_processing, DataProcessingTools)
        assert isinstance(u.configuration, ConfigurationManager)
        assert isinstance(u.file_manager, FileManager)
        assert isinstance(u.mathematics, MathematicalUtilities)
        assert isinstance(u.helpers, HelperFunctions)

    def test_process_time_series_default_operations(self) -> None:
        data = [0.0, 1.0, 2.0, 3.0, 4.0]
        result = Utilities({}).process_time_series(data)
        assert result["original"] is data
        assert result["normalized"] == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])
        assert result["smoothed"] == pytest.approx([1.0, 1.5, 2.0, 2.5, 3.0])
        assert result["statistics"]["mean"] == pytest.approx(2.0)

    def test_process_time_series_selected_operations(self) -> None:
        result = Utilities({}).process_time_series([1.0, 3.0], ["statistics"])
        assert set(result) == {"original", "statistics"}
        assert result["statistics"]["count"] == 2

    def test_load_and_merge_configs(self, tmp_path: Path) -> None:
        base = tmp_path / "base.json"
        override = tmp_path / "override.yaml"
        base.write_text(json.dumps({"a": 1, "n": {"x": 1}}))
        override.write_text("n:\n  y: 2\n")
        merged = Utilities({}).load_and_merge_configs(base, override)
        assert merged == {"a": 1, "n": {"x": 1, "y": 2}}

    def test_backup_and_save_new_file(self, tmp_path: Path) -> None:
        target = tmp_path / "data.json"
        assert Utilities({}).backup_and_save({"v": 1}, target) is True
        assert json.loads(target.read_text()) == {"v": 1}
        assert not (tmp_path / "data.json.backup").exists()

    def test_backup_and_save_existing_file(self, tmp_path: Path) -> None:
        target = tmp_path / "data.json"
        target.write_text('{"v": 0}')
        assert Utilities({}).backup_and_save({"v": 1}, target, ".bak") is True
        assert (tmp_path / "data.json.bak").read_text() == '{"v": 0}'
        assert json.loads(target.read_text()) == {"v": 1}

    def test_backup_failure_still_saves(self, tmp_path: Path) -> None:
        target = tmp_path / "data.json"
        target.write_text('{"v": 0}')
        # Occupy the backup path with a directory so the backup write fails.
        (tmp_path / "data.json.backup").mkdir()
        assert Utilities({}).backup_and_save({"v": 2}, target) is True
        assert json.loads(target.read_text()) == {"v": 2}
        assert (tmp_path / "data.json.backup").is_dir()
