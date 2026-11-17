"""Coverage-focused behaviour tests for ``llm.conversations``."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pytest

from active_inference.llm import conversations as conv_mod
from active_inference.llm.conversations import (
    ActiveInferenceTemplates,
    Conversation,
    ConversationManager,
    ConversationTemplate,
    Message,
)


@pytest.fixture
def manager(tmp_path: Path) -> ConversationManager:
    return ConversationManager(storage_path=tmp_path / "convs")


class TestConversation:
    def test_title_generation_and_truncation(self) -> None:
        conv = Conversation(id="c")
        conv.add_message("user", "  short question  ")
        assert conv.title == "short question"
        long_conv = Conversation(id="d")
        long_conv.add_message("user", "x" * 60)
        assert long_conv.title == "x" * 50 + "..."
        sys_first = Conversation(id="e")
        sys_first.add_message("system", "rules")
        sys_first.add_message("user", "hello")
        assert sys_first.title == ""
        titled = Conversation(id="f", title="Keep")
        titled.add_message("user", "other")
        assert titled.title == "Keep"

    def test_get_messages_limit(self) -> None:
        conv = Conversation(id="c")
        for i in range(4):
            conv.add_message("user", f"m{i}", {"i": i})
        assert [m.content for m in conv.get_messages(2)] == ["m2", "m3"]
        assert len(conv.get_messages()) == 4
        assert conv.messages[1].metadata == {"i": 1}

    def test_context_messages_respect_token_budget(self) -> None:
        conv = Conversation(id="c")
        conv.add_message("system", "one two three")  # 3.9 tokens
        conv.add_message("user", "four five")  # 2.6 tokens
        conv.add_message("assistant", "six")  # would exceed 7
        conv.add_message("user", "x")
        context = conv.get_context_messages(max_tokens=7)
        assert [m.content for m in context] == ["one two three", "four five"]

    def test_clear_and_summary(self) -> None:
        conv = Conversation(id="c", title="T", metadata={"k": 1})
        conv.add_message("user", "hi")
        before = conv.updated_at
        conv.clear_messages()
        assert conv.messages == []
        assert conv.updated_at >= before
        summary = conv.get_summary()
        assert summary["id"] == "c"
        assert summary["message_count"] == 0
        assert summary["metadata"] == {"k": 1}
        assert datetime.fromisoformat(summary["updated_at"]) == conv.updated_at

    def test_dict_round_trip(self) -> None:
        conv = Conversation(id="c", title="T", metadata={"a": 1})
        conv.add_message("user", "hi", {"src": "test"})
        restored = Conversation.from_dict(json.loads(json.dumps(conv.to_dict())))
        assert restored == conv

    def test_message_from_dict_default_metadata(self) -> None:
        msg = Message.from_dict(
            {"role": "user", "content": "x", "timestamp": "2026-01-02T03:04:05"}
        )
        assert msg.metadata == {}
        assert msg.timestamp == datetime(2026, 1, 2, 3, 4, 5)


class TestManagerPersistence:
    def test_create_persists_and_reloads(
        self, manager: ConversationManager, tmp_path: Path
    ) -> None:
        conv = manager.create_conversation("First", {"topic": "fep"})
        assert manager.add_message(conv.id, "user", "What is FEP?") is True
        saved = json.loads((tmp_path / "convs" / f"{conv.id}.json").read_text())
        assert saved["messages"][0]["content"] == "What is FEP?"
        reloaded = ConversationManager(storage_path=tmp_path / "convs")
        assert reloaded.get_conversation(conv.id) == conv

    def test_add_message_to_unknown(self, manager: ConversationManager) -> None:
        assert manager.add_message("nope", "user", "x") is False

    def test_corrupt_file_is_skipped(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        storage = tmp_path / "convs"
        storage.mkdir()
        (storage / "bad.json").write_text("{oops")
        with caplog.at_level("ERROR"):
            mgr = ConversationManager(storage_path=storage)
        assert mgr.conversations == {}
        assert "Failed to load conversation" in caplog.text

    def test_storage_listing_failure_is_logged(
        self, manager: ConversationManager, caplog: pytest.LogCaptureFixture
    ) -> None:
        def broken_glob(pattern: str) -> Any:
            raise PermissionError("denied")

        manager.storage_path = SimpleNamespace(glob=broken_glob)  # type: ignore
        with caplog.at_level("ERROR"):
            manager._load_conversations()
        assert "Failed to load conversations: denied" in caplog.text

    def test_save_failure_is_logged(
        self,
        manager: ConversationManager,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        manager.storage_path = tmp_path / "missing"
        with caplog.at_level("ERROR"):
            conv = manager.create_conversation("unsaved")
        assert manager.get_conversation(conv.id) is conv
        assert f"Failed to save conversation {conv.id}" in caplog.text
        assert not (tmp_path / "missing").exists()

    def test_delete(self, manager: ConversationManager, tmp_path: Path) -> None:
        conv = manager.create_conversation("x")
        path = tmp_path / "convs" / f"{conv.id}.json"
        assert path.exists()
        assert manager.delete_conversation(conv.id) is True
        assert not path.exists()
        assert manager.delete_conversation(conv.id) is False
        # In-memory only conversation (no file on disk) still deletes.
        manager.conversations["mem"] = Conversation(id="mem")
        assert manager.delete_conversation("mem") is True


class TestManagerQueries:
    def test_search_by_title_and_content(self, manager: ConversationManager) -> None:
        a = manager.create_conversation("Entropy chat")
        b = manager.create_conversation("Other")
        manager.add_message(b.id, "user", "talk about ENTROPY please")
        manager.add_message(b.id, "user", "and entropy again")
        manager.create_conversation("Unrelated")
        a.updated_at = datetime(2020, 1, 1)
        b.updated_at = datetime(2021, 1, 1)
        results = manager.search_conversations("entropy")
        assert [c.id for c in results] == [b.id, a.id]
        assert [c.id for c in manager.search_conversations("entropy", limit=1)] == [
            b.id
        ]
        assert manager.list_conversations()[0] is a

    def test_context_for_api(self, manager: ConversationManager) -> None:
        conv = manager.create_conversation("x")
        manager.add_message(conv.id, "system", "be brief")
        manager.add_message(conv.id, "user", "hi")
        assert manager.get_conversation_context(conv.id) == [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
        ]
        assert manager.get_conversation_context("nope") == []

    def test_export(self, manager: ConversationManager) -> None:
        conv = manager.create_conversation("Talk")
        manager.add_message(conv.id, "user", "hello")
        conv.messages[0].timestamp = datetime(2026, 5, 6, 7, 8, 9)
        as_json = json.loads(manager.export_conversation(conv.id, "JSON"))
        assert as_json["title"] == "Talk"
        md = manager.export_conversation(conv.id, "markdown")
        assert md == "# Talk\n\n## USER - 2026-05-06 07:08:09\n\nhello\n"
        assert manager.export_conversation("nope") is None
        with pytest.raises(ValueError, match="Unsupported export format: pdf"):
            manager.export_conversation(conv.id, "pdf")

    def test_cleanup_old(self, manager: ConversationManager) -> None:
        old = manager.create_conversation("old")
        fresh = manager.create_conversation("fresh")
        old.updated_at = datetime.now() - timedelta(days=40)
        assert manager.cleanup_old_conversations(days=30) == 1
        assert manager.get_conversation(old.id) is None
        assert manager.get_conversation(fresh.id) is fresh

    def test_statistics(self, manager: ConversationManager) -> None:
        empty = manager.get_statistics()
        assert empty["total_conversations"] == 0
        assert empty["oldest_conversation"] is None
        assert empty["newest_conversation"] is None
        a = manager.create_conversation("a")
        b = manager.create_conversation("b")
        manager.add_message(a.id, "user", "x")
        manager.add_message(a.id, "assistant", "y")
        manager.add_message(b.id, "user", "z")
        a.created_at = datetime(2020, 1, 1)
        b.created_at = datetime(2025, 1, 1)
        b.updated_at = datetime.now() - timedelta(days=30)
        stats = manager.get_statistics()
        assert stats["total_conversations"] == 2
        assert stats["total_messages"] == 3
        assert stats["role_distribution"] == {"user": 2, "assistant": 1}
        assert stats["recent_conversations"] == 1
        assert stats["oldest_conversation"] == datetime(2020, 1, 1)
        assert stats["newest_conversation"] == datetime(2025, 1, 1)


class TestTemplates:
    def test_template_with_manager(self, manager: ConversationManager) -> None:
        tpl = ConversationTemplate(
            "tpl",
            "be helpful",
            initial_messages=[{"role": "assistant", "content": "Hi!"}],
            metadata={"kind": "demo"},
        )
        conv = tpl.create_conversation(manager=manager)
        assert conv.title == "tpl"
        assert conv.metadata == {"kind": "demo"}
        assert [(m.role, m.content) for m in conv.messages] == [
            ("system", "be helpful"),
            ("assistant", "Hi!"),
        ]
        assert manager.get_conversation(conv.id) is conv
        titled = ConversationTemplate("t2", "").create_conversation(
            "Custom", manager=manager
        )
        assert titled.title == "Custom"
        assert titled.messages == []

    def test_template_without_manager_uses_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        created = []
        real = conv_mod.ConversationManager

        def default_manager(storage_path: Optional[Path] = None) -> Any:
            mgr = real(storage_path=storage_path or tmp_path / "default")
            created.append(mgr)
            return mgr

        monkeypatch.setattr(conv_mod, "ConversationManager", default_manager)
        conv = ConversationTemplate("tpl", "sys").create_conversation()
        assert len(created) == 1
        assert created[0].get_conversation(conv.id) is conv
        assert (tmp_path / "default" / f"{conv.id}.json").exists()

    @pytest.mark.parametrize(
        ("factory", "name", "kind"),
        [
            (
                ActiveInferenceTemplates.get_explanation_template,
                "active_inference_explanation",
                "explanation",
            ),
            (
                ActiveInferenceTemplates.get_research_template,
                "research_discussion",
                "research",
            ),
            (
                ActiveInferenceTemplates.get_implementation_template,
                "implementation_help",
                "implementation",
            ),
            (
                ActiveInferenceTemplates.get_education_template,
                "education_development",
                "education",
            ),
        ],
    )
    def test_predefined_templates(self, factory: Any, name: str, kind: str) -> None:
        tpl = factory()
        assert tpl.name == name
        assert tpl.metadata == {"type": kind, "domain": "active_inference"}
        assert "Active Inference" in tpl.system_prompt
        assert tpl.initial_messages == []
