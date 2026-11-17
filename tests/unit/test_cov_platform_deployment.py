"""Behavioural tests for active_inference.platform.deployment.

psutil and time.sleep are patched at the module boundary so the tests are
hermetic and do not block on real CPU sampling or simulated deploy delays.
"""

import time
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from active_inference.platform import deployment as dep
from active_inference.platform.deployment import (
    Deployment,
    DeploymentAutomation,
    DeploymentManager,
    DeploymentStatus,
    HealthMonitoring,
    MonitoringTools,
    ServiceOrchestrator,
    ServiceStatus,
)


@pytest.fixture
def no_sleep(monkeypatch):
    calls = []
    clock = SimpleNamespace(time=time.time, sleep=calls.append)
    monkeypatch.setattr(dep, "time", clock)
    return calls


@pytest.fixture
def fake_psutil(monkeypatch):
    """Replace psutil probes with deterministic values (adjustable per test)."""
    state = {"cpu": 10.0, "mem": 20.0, "disk": 30.0, "conns": 4, "cpu_calls": []}

    def cpu_percent(interval=None):
        state["cpu_calls"].append(interval)
        return state["cpu"]

    monkeypatch.setattr(dep.psutil, "cpu_percent", cpu_percent)
    monkeypatch.setattr(
        dep.psutil, "virtual_memory", lambda: SimpleNamespace(percent=state["mem"])
    )
    monkeypatch.setattr(
        dep.psutil, "disk_usage", lambda path: SimpleNamespace(percent=state["disk"])
    )
    monkeypatch.setattr(
        dep.psutil, "net_connections", lambda: [object()] * state["conns"]
    )
    return state


class TestServiceOrchestrator:
    def test_register_service_reads_config(self):
        orch = ServiceOrchestrator({"services": {"x": {}}})
        assert orch.service_configs == {"x": {}}
        cfg = {"port": 8080, "health_endpoint": "/h", "dependencies": ["db"]}
        assert orch.register_service("api", cfg) is True
        info = orch.services["api"]
        assert info.status == ServiceStatus.STOPPED
        assert info.port == 8080
        assert info.health_endpoint == "/h"
        assert info.dependencies == ["db"]
        assert info.config is cfg

    def test_register_service_with_invalid_config_fails(self):
        orch = ServiceOrchestrator({})
        assert orch.register_service("api", "not-a-dict") is False
        assert "api" not in orch.services

    def test_start_unregistered_service_fails(self):
        assert ServiceOrchestrator({}).start_service("ghost") is False

    def test_start_respects_dependencies(self):
        orch = ServiceOrchestrator({})
        orch.register_service("db", {})
        orch.register_service("api", {"dependencies": ["db"]})
        orch.register_service("web", {"dependencies": ["missing"]})

        assert orch.start_service("api") is False
        assert orch.services["api"].status == ServiceStatus.STOPPED

        assert orch.start_service("web") is False

        assert orch.start_service("db") is True
        assert orch.start_service("api") is True
        assert orch.services["api"].status == ServiceStatus.RUNNING
        assert isinstance(orch.services["api"].start_time, datetime)

    def test_start_marks_service_error_on_exception(self):
        orch = ServiceOrchestrator({})
        orch.register_service("bad", {"dependencies": 5})  # not iterable
        assert orch.start_service("bad") is False
        assert orch.services["bad"].status == ServiceStatus.ERROR

    def test_stop_service(self):
        orch = ServiceOrchestrator({})
        assert orch.stop_service("ghost") is False
        orch.register_service("db", {})
        orch.start_service("db")
        orch.services["db"].pid = 123
        assert orch.stop_service("db") is True
        assert orch.services["db"].status == ServiceStatus.STOPPED
        assert orch.services["db"].pid is None

    def test_stop_service_invalid_name_fails(self):
        assert ServiceOrchestrator({}).stop_service(["unhashable"]) is False

    def test_get_service_status_single_and_all(self):
        orch = ServiceOrchestrator({})
        orch.register_service("db", {"port": 5432})
        orch.register_service("cache", {})
        orch.start_service("db")
        orch.services["db"].start_time = datetime.now() - timedelta(seconds=90)

        single = orch.get_service_status("db")
        assert single["name"] == "db"
        assert single["status"] == "running"
        assert single["port"] == 5432
        assert 89 <= single["uptime"] <= 91

        assert orch.get_service_status("ghost") == {}

        everything = orch.get_service_status()["services"]
        assert set(everything) == {"db", "cache"}
        assert everything["cache"] == {
            "status": "stopped",
            "pid": None,
            "port": None,
            "uptime": 0,
        }
        assert everything["db"]["uptime"] >= 89

        assert orch.get_service_status("cache")["uptime"] == 0

    def test_check_service_health(self):
        orch = ServiceOrchestrator({})
        assert orch.check_service_health("ghost") == {
            "status": "unknown",
            "message": "Service not found",
        }
        orch.register_service("db", {})
        assert orch.check_service_health("db") == {
            "status": "unhealthy",
            "message": "Service status: stopped",
        }
        orch.start_service("db")
        assert orch.check_service_health("db") == {
            "status": "healthy",
            "message": "Service is running",
        }

    def test_check_service_health_reports_errors(self):
        orch = ServiceOrchestrator({})
        orch.register_service("db", {})
        orch.services["db"].status = "corrupt"  # no .value attribute
        result = orch.check_service_health("db")
        assert result["status"] == "error"
        assert "value" in result["message"]

    def test_get_service_dependencies(self):
        orch = ServiceOrchestrator({})
        orch.register_service("api", {"dependencies": ["db", "cache"]})
        assert orch.get_service_dependencies("api") == ["db", "cache"]
        assert orch.get_service_dependencies("ghost") == []


class TestHealthMonitoring:
    def test_record_metric_and_history_limit(self):
        hm = HealthMonitoring({"max_metrics_history": 3})
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            hm.record_metric("latency", v, timestamp=0.0)
        assert hm.metrics["latency"] == [3.0, 4.0, 5.0]
        assert hm.get_health_history("latency") == [3.0, 4.0, 5.0]
        assert hm.get_health_history("latency", limit=2) == [4.0, 5.0]
        assert hm.get_health_history("missing") == []

    def test_metric_average(self):
        hm = HealthMonitoring({})
        assert hm.max_metrics_history == 1000
        assert hm.get_metric_average("x") is None
        hm.metrics["empty"] = []
        assert hm.get_metric_average("empty") is None
        for v in [2.0, 4.0, 6.0, 8.0]:
            hm.record_metric("x", v)
        assert hm.get_metric_average("x") == 5.0
        assert hm.get_metric_average("x", window=2) == 7.0

    def test_thresholds_no_alerts(self, fake_psutil):
        hm = HealthMonitoring({})
        assert hm.check_health_thresholds() == []
        assert hm.alerts == []
        assert fake_psutil["cpu_calls"] == [1]

    def test_thresholds_default_alerts(self, fake_psutil):
        fake_psutil["cpu"] = 95.0
        fake_psutil["mem"] = 90.0
        hm = HealthMonitoring({})
        alerts = hm.check_health_thresholds()
        assert [a["type"] for a in alerts] == ["cpu_high", "memory_high"]
        assert alerts[0]["message"] == "CPU usage is 95.0%"
        assert alerts[1]["message"] == "Memory usage is 90.0%"
        assert all(a["severity"] == "warning" for a in alerts)
        assert hm.alerts == alerts

    def test_thresholds_from_config(self, fake_psutil):
        fake_psutil["cpu"] = 50.0
        fake_psutil["mem"] = 20.0
        hm = HealthMonitoring({"thresholds": {"cpu_usage": 40.0, "memory_usage": 90.0}})
        alerts = hm.check_health_thresholds()
        assert [a["type"] for a in alerts] == ["cpu_high"]

    def test_system_health_records_metrics(self, fake_psutil):
        hm = HealthMonitoring({})
        info = hm.get_system_health()
        assert info["cpu_usage"] == 10.0
        assert info["memory_usage"] == 20.0
        assert info["disk_usage"] == 30.0
        assert info["network_connections"] == 4
        assert info["alerts"] == []
        assert isinstance(info["timestamp"], datetime)
        assert hm.metrics == {
            "cpu_usage": [10.0],
            "memory_usage": [20.0],
            "disk_usage": [30.0],
        }


class TestDeploymentAutomation:
    def test_successful_deploy(self, no_sleep, monkeypatch):
        monkeypatch.setattr(dep.time, "time", lambda: 1700000000.5)
        da = DeploymentAutomation({})
        dep_id = da.deploy_service("api", "1.2.3", ["fix"])
        assert dep_id == "deploy_api_1700000000"
        assert no_sleep == [1]

        status = da.get_deployment_status(dep_id)
        assert status["service_name"] == "api"
        assert status["status"] == "successful"
        assert status["version"] == "1.2.3"
        assert status["changes"] == ["fix"]
        assert status["error_message"] is None
        assert status["end_time"] is not None
        datetime.fromisoformat(status["start_time"])

    def test_deploy_failure_is_recorded(self, monkeypatch):
        def boom(_):
            raise RuntimeError("disk full")

        monkeypatch.setattr(dep, "time", SimpleNamespace(time=time.time, sleep=boom))
        da = DeploymentAutomation({})
        dep_id = da.deploy_service("api", "2.0")
        record = da.deployments[dep_id]
        assert record.status == DeploymentStatus.FAILED
        assert record.error_message == "disk full"
        assert record.changes == []
        assert record.end_time is not None

    def test_rollback(self, no_sleep):
        da = DeploymentAutomation({})
        assert da.rollback_deployment("missing") is False
        dep_id = da.deploy_service("api", "1.0")
        assert da.rollback_deployment(dep_id) is True
        assert da.get_deployment_status(dep_id)["status"] == "rolled_back"

    def test_rollback_failure_returns_false(self, monkeypatch):
        da = DeploymentAutomation({})
        da.deployments["d1"] = Deployment(
            id="d1",
            service_name="api",
            status=DeploymentStatus.SUCCESSFUL,
            start_time=datetime(2026, 1, 1),
        )

        class BrokenClock:
            @staticmethod
            def now():
                raise OSError("clock unavailable")

        monkeypatch.setattr(dep, "datetime", BrokenClock)
        assert da.rollback_deployment("d1") is False

    def test_status_for_unknown_and_pending(self):
        da = DeploymentAutomation({})
        assert da.get_deployment_status("missing") is None
        da.deployments["p"] = Deployment(
            id="p",
            service_name="api",
            status=DeploymentStatus.PENDING,
            start_time=datetime(2026, 1, 1, 12, 0),
        )
        status = da.get_deployment_status("p")
        assert status["status"] == "pending"
        assert status["end_time"] is None
        assert status["start_time"] == "2026-01-01T12:00:00"

    def test_history_sorted_filtered_and_limited(self):
        da = DeploymentAutomation({})
        for i, svc in enumerate(["api", "web", "api", "api"]):
            da.deployments[f"d{i}"] = Deployment(
                id=f"d{i}",
                service_name=svc,
                status=DeploymentStatus.SUCCESSFUL,
                start_time=datetime(2026, 1, 1) + timedelta(hours=i),
            )
        assert [d["id"] for d in da.get_deployment_history()] == [
            "d3",
            "d2",
            "d1",
            "d0",
        ]
        assert [d["id"] for d in da.get_deployment_history("api", limit=2)] == [
            "d3",
            "d2",
        ]
        assert da.get_deployment_history("none") == []


class TestMonitoringTools:
    def test_log_event_and_logger_dispatch(self, caplog):
        mt = MonitoringTools({})
        assert mt.max_logs == 10000
        with caplog.at_level("DEBUG", logger=dep.logger.name):
            mt.log_event("ERROR", "db down", service="db", details={"code": 7})
            mt.log_event("NOTICE", "custom level", service="db")
        assert mt.logs[0]["level"] == "ERROR"
        assert mt.logs[0]["details"] == {"code": 7}
        assert mt.logs[1]["details"] == {}
        levels = [(r.levelname, r.getMessage()) for r in caplog.records]
        assert ("ERROR", "[db] db down") in levels
        # unknown levels fall back to logger.info
        assert ("INFO", "[db] custom level") in levels

    def test_log_size_limit(self):
        mt = MonitoringTools({"max_logs": 2})
        for i in range(4):
            mt.log_event("INFO", f"m{i}")
        assert [entry["message"] for entry in mt.logs] == ["m2", "m3"]

    def test_get_logs_filters(self):
        mt = MonitoringTools({})
        mt.log_event("INFO", "a", service="api")
        mt.log_event("ERROR", "b", service="api")
        mt.log_event("ERROR", "c", service="db")
        assert [x["message"] for x in mt.get_logs(level="ERROR")] == ["b", "c"]
        assert [x["message"] for x in mt.get_logs(service="api")] == ["a", "b"]
        assert [x["message"] for x in mt.get_logs("ERROR", "db")] == ["c"]
        assert [x["message"] for x in mt.get_logs(limit=1)] == ["c"]

    def test_service_metrics_and_health_report(self):
        mt = MonitoringTools({})
        mt.log_event("INFO", "a", service="api")
        mt.log_event("ERROR", "b", service="api")
        mt.log_event("WARNING", "c", service="api")
        mt.log_event("DEBUG", "d", service="db")
        assert mt.get_service_metrics() == {
            "api": {"errors": 1, "warnings": 1, "info": 1},
            "db": {"errors": 0, "warnings": 0, "info": 0},
        }
        report = mt.generate_health_report()
        assert report["system_status"] == "operational"
        assert report["recommendations"] == []
        assert [x["message"] for x in report["recent_alerts"]] == ["c"]
        assert report["log_summary"]["api"]["errors"] == 1


class TestDeploymentManager:
    def test_subcomponent_configs(self):
        dm = DeploymentManager(
            {
                "health_monitoring": {"max_metrics_history": 5},
                "monitoring_tools": {"max_logs": 9},
                "orchestrator": {"services": {"a": {}}},
            }
        )
        assert dm.health_monitoring.max_metrics_history == 5
        assert dm.monitoring_tools.max_logs == 9
        assert dm.orchestrator.service_configs == {"a": {}}
        assert dm.deployment_automation.config == {}

    def test_deploy_platform(self, no_sleep):
        dm = DeploymentManager({})
        result = dm.deploy_platform("9.9")
        status = result["status"]
        assert status["id"] == result["deployment_id"]
        assert status["service_name"] == "platform"
        assert status["version"] == "9.9"
        assert status["status"] == "successful"
        assert len(status["changes"]) == 3
        assert dm.deploy_platform()["status"]["version"] == "latest"

    def test_platform_status(self, fake_psutil, no_sleep):
        dm = DeploymentManager({})
        dm.orchestrator.register_service("search", {})
        dm.deployment_automation.deploy_service("search", "1.0")
        dm.monitoring_tools.log_event("INFO", "hello")
        status = dm.get_platform_status()
        assert set(status["services"]["services"]) == {"search"}
        assert status["health"]["cpu_usage"] == 10.0
        assert [d["service_name"] for d in status["deployments"]] == ["search"]
        assert [log["message"] for log in status["logs"]] == ["hello"]

    def test_start_and_stop_platform_services(self):
        dm = DeploymentManager({})
        assert dm.start_platform_services() == {
            "knowledge_graph": "failed",
            "search": "failed",
            "collaboration": "failed",
            "platform": "failed",
        }

        dm.orchestrator.register_service("knowledge_graph", {})
        dm.orchestrator.register_service(
            "search", {"dependencies": ["knowledge_graph"]}
        )
        dm.orchestrator.register_service("collaboration", {})
        dm.orchestrator.register_service("platform", {"dependencies": ["search"]})
        assert dm.start_platform_services() == {
            "knowledge_graph": "started",
            "search": "started",
            "collaboration": "started",
            "platform": "started",
        }

        del dm.orchestrator.services["collaboration"]
        assert dm.stop_platform_services() == {
            "platform": "stopped",
            "collaboration": "failed",
            "search": "stopped",
            "knowledge_graph": "stopped",
        }
        assert dm.orchestrator.services["search"].status == ServiceStatus.STOPPED

    def test_platform_health_report(self):
        dm = DeploymentManager({})
        dm.monitoring_tools.log_event("WARNING", "slow", service="search")
        report = dm.get_platform_health_report()
        assert report["log_summary"] == {
            "search": {"errors": 0, "warnings": 1, "info": 0}
        }
        assert report["recent_alerts"][0]["message"] == "slow"
