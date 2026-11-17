"""Behavioural tests for active_inference.applications.integrations."""

from typing import Any

from active_inference.applications.integrations import (
    APIConnector,
    IntegrationManager,
    IntegrationType,
)


def test_integration_type_values() -> None:
    assert IntegrationType("database") is IntegrationType.DATABASE
    assert {t.value for t in IntegrationType} == {
        "database",
        "api",
        "message_queue",
        "file_system",
        "web_service",
        "iot_device",
    }


def test_connector_defaults_and_independent_headers() -> None:
    first = APIConnector(name="a", base_url="http://a")
    second = APIConnector(name="b", base_url="http://b")
    assert first.auth_type == "none"
    assert first.timeout == 30
    assert first.headers == {}
    first.headers["X"] = "1"
    assert second.headers == {}


def test_register_and_reject_duplicate() -> None:
    manager = IntegrationManager({"k": 1})
    connector = APIConnector(name="svc", base_url="http://svc", auth_type="bearer")
    assert manager.config == {"k": 1}
    assert manager.register_api_connector(connector) is True
    assert manager.register_api_connector(APIConnector("svc", "http://other")) is False
    assert manager.connectors["svc"] is connector


def test_connect_unknown_connector() -> None:
    manager = IntegrationManager({})
    assert manager.connect_to_api("missing") is False
    assert manager.active_connections == {}


def test_connect_query_disconnect_lifecycle() -> None:
    manager = IntegrationManager({})
    connector = APIConnector(name="svc", base_url="http://svc", timeout=5)
    manager.register_api_connector(connector)

    # Query before connecting fails.
    assert manager.query_api("svc", "/items") is None

    assert manager.connect_to_api("svc") is True
    connection = manager.active_connections["svc"]
    assert connection["connector"] is connector
    assert connection["status"] == "connected"

    response = manager.query_api("svc", "/items", {"page": 2})
    assert response == {
        "endpoint": "/items",
        "params": {"page": 2},
        "response": "mock_data",
        "status": "success",
    }
    assert manager.query_api("svc", "/x")["params"] is None  # type: ignore[index]

    listing = manager.list_integrations()
    assert listing == {
        "connectors": {
            "svc": {
                "name": "svc",
                "base_url": "http://svc",
                "auth_type": "none",
                "timeout": 5,
            }
        },
        "active_connections": ["svc"],
    }

    assert manager.disconnect_api("svc") is True
    assert manager.disconnect_api("svc") is False
    assert manager.list_integrations()["active_connections"] == []


def test_connect_failure_is_reported() -> None:
    class FailingStore(dict):  # type: ignore[type-arg]
        def __setitem__(self, key: Any, value: Any) -> None:
            raise RuntimeError("store unavailable")

    manager = IntegrationManager({})
    manager.register_api_connector(APIConnector(name="svc", base_url="http://svc"))
    manager.active_connections = FailingStore()
    assert manager.connect_to_api("svc") is False
    assert "svc" not in manager.active_connections
