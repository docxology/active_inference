"""Behavioural tests for active_inference.platform.collaboration."""

from datetime import datetime, timedelta

import pytest

from active_inference.platform.collaboration import (
    Activity,
    ActivityTracker,
    CollaborationManager,
    Permission,
    User,
    UserManagement,
    UserRole,
    Workspace,
    WorkspaceManager,
)

UNHASHABLE_ID = ["not", "hashable"]


def make_user(user_id="u1", username="alice", role=UserRole.RESEARCHER):
    return User(id=user_id, username=username, email=f"{username}@x.org", role=role)


def make_activity(idx, user_id="u1", workspace_id="w1", action="edit", ts=None):
    return Activity(
        id=f"a{idx}",
        user_id=user_id,
        workspace_id=workspace_id,
        action=action,
        resource_type="doc",
        resource_id=f"r{idx}",
        timestamp=ts or datetime(2026, 1, 1) + timedelta(minutes=idx),
    )


class TestUserPermissions:
    @pytest.mark.parametrize(
        "role, allowed",
        [
            (UserRole.ADMIN, set(Permission)),
            (
                UserRole.RESEARCHER,
                {Permission.READ, Permission.WRITE, Permission.SHARE},
            ),
            (UserRole.EDUCATOR, {Permission.READ, Permission.WRITE, Permission.SHARE}),
            (UserRole.STUDENT, {Permission.READ}),
            (UserRole.GUEST, {Permission.READ}),
        ],
    )
    def test_role_permission_matrix(self, role, allowed):
        user = make_user(role=role)
        granted = {p for p in Permission if user.has_permission(p)}
        assert granted == allowed

    def test_unknown_role_has_no_permissions(self):
        user = make_user()
        user.role = "superuser"  # not a UserRole member
        assert not any(user.has_permission(p) for p in Permission)


class TestWorkspaceDataclass:
    def test_add_member_updates_list_and_timestamp(self):
        ws = Workspace(id="w", name="n", description="d", owner_id="owner")
        ws.last_modified = datetime(2000, 1, 1)
        assert ws.add_member("bob") is True
        assert ws.member_ids == ["bob"]
        assert ws.last_modified > datetime(2000, 1, 1)

    def test_add_existing_member_is_rejected(self):
        ws = Workspace(
            id="w", name="n", description="d", owner_id="o", member_ids=["b"]
        )
        stamp = ws.last_modified
        assert ws.add_member("b") is False
        assert ws.member_ids == ["b"]
        assert ws.last_modified == stamp

    def test_remove_member(self):
        ws = Workspace(
            id="w", name="n", description="d", owner_id="o", member_ids=["b"]
        )
        assert ws.remove_member("b") is True
        assert ws.member_ids == []
        assert ws.remove_member("b") is False

    def test_is_member_includes_owner(self):
        ws = Workspace(
            id="w", name="n", description="d", owner_id="o", member_ids=["b"]
        )
        assert ws.is_member("o")
        assert ws.is_member("b")
        assert not ws.is_member("c")


class TestUserManagement:
    def test_add_get_and_duplicate(self):
        um = UserManagement({})
        user = make_user()
        assert um.add_user(user) is True
        assert um.get_user("u1") is user
        assert um.add_user(make_user(username="other")) is False
        assert um.get_user("u1").username == "alice"
        assert um.get_user("missing") is None

    def test_update_user(self):
        um = UserManagement({})
        assert um.update_user(make_user()) is False
        um.add_user(make_user())
        updated = make_user(username="alice2")
        assert um.update_user(updated) is True
        assert um.get_user("u1").username == "alice2"

    def test_delete_user(self):
        um = UserManagement({})
        assert um.delete_user("u1") is False
        um.add_user(make_user())
        assert um.delete_user("u1") is True
        assert um.get_user("u1") is None

    def test_invalid_ids_are_reported_as_failures(self, caplog):
        um = UserManagement({})
        bad = make_user(user_id=UNHASHABLE_ID)
        assert um.add_user(bad) is False
        assert um.update_user(bad) is False
        assert um.delete_user(UNHASHABLE_ID) is False
        assert um.users == {}
        errors = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
        assert any("Error adding user" in m for m in errors)
        assert any("Error updating user" in m for m in errors)
        assert any("Error deleting user" in m for m in errors)

    def test_authenticate_validate_logout(self):
        um = UserManagement({})
        user = make_user(role=UserRole.EDUCATOR)
        user.last_active = datetime(2000, 1, 1)
        um.add_user(user)

        assert um.authenticate_user("nobody", "pw") is None

        session_id = um.authenticate_user("alice", "pw")
        assert isinstance(session_id, str)
        info = um.validate_session(session_id)
        assert info["user_id"] == "u1"
        assert info["username"] == "alice"
        assert info["role"] == "educator"
        assert isinstance(info["created_at"], datetime)
        assert user.last_active > datetime(2000, 1, 1)

        assert um.logout_user(session_id) is True
        assert um.validate_session(session_id) is None
        assert um.logout_user(session_id) is False

    def test_user_statistics(self):
        um = UserManagement({})
        um.add_user(make_user("u1", "a", UserRole.STUDENT))
        um.add_user(make_user("u2", "b", UserRole.STUDENT))
        um.add_user(make_user("u3", "c", UserRole.ADMIN))
        um.authenticate_user("a", "pw")
        assert um.get_user_statistics() == {
            "total_users": 3,
            "active_sessions": 1,
            "role_distribution": {"student": 2, "admin": 1},
        }


class TestWorkspaceManager:
    def make_ws(self, ws_id="w1", owner="o", members=None):
        return Workspace(
            id=ws_id,
            name=f"name-{ws_id}",
            description="d",
            owner_id=owner,
            member_ids=list(members or []),
        )

    def test_create_maps_owner_and_members(self):
        wm = WorkspaceManager({})
        ws = self.make_ws(members=["m1"])
        assert wm.create_workspace(ws) is True
        assert wm.get_workspace("w1") is ws
        assert wm.user_workspaces == {"o": {"w1"}, "m1": {"w1"}}
        assert wm.create_workspace(self.make_ws()) is False
        assert wm.get_workspace("nope") is None

    def test_update_workspace_reconciles_membership(self):
        wm = WorkspaceManager({})
        assert wm.update_workspace(self.make_ws()) is False

        wm.create_workspace(self.make_ws(members=["m1", "m2"]))
        replacement = self.make_ws(members=["m2", "m3"])
        assert wm.update_workspace(replacement) is True
        assert wm.get_workspace("w1") is replacement
        assert "m1" not in wm.user_workspaces
        assert wm.user_workspaces["m2"] == {"w1"}
        assert wm.user_workspaces["m3"] == {"w1"}

    def test_delete_workspace_clears_mappings(self):
        wm = WorkspaceManager({})
        assert wm.delete_workspace("w1") is False
        wm.create_workspace(self.make_ws(members=["m1"]))
        wm.create_workspace(self.make_ws("w2", owner="o"))
        assert wm.delete_workspace("w1") is True
        assert "w1" not in wm.workspaces
        assert wm.user_workspaces == {"o": {"w2"}}

    def test_add_member_permissions(self):
        wm = WorkspaceManager({})
        assert wm.add_member("missing", "x", "o") is False

        wm.create_workspace(self.make_ws(members=["m1"]))
        # outsider cannot invite
        assert wm.add_member("w1", "x", "outsider") is False
        # owner and members can invite
        assert wm.add_member("w1", "x", "o") is True
        assert wm.add_member("w1", "y", "m1") is True
        assert wm.get_workspace("w1").member_ids == ["m1", "x", "y"]
        assert wm.user_workspaces["y"] == {"w1"}
        # duplicate invite is rejected
        assert wm.add_member("w1", "x", "o") is False

    def test_remove_member_owner_only(self):
        wm = WorkspaceManager({})
        assert wm.remove_member("missing", "m1", "o") is False

        wm.create_workspace(self.make_ws(members=["m1", "m2"]))
        assert wm.remove_member("w1", "m2", "m1") is False
        assert wm.remove_member("w1", "m2", "o") is True
        assert wm.get_workspace("w1").member_ids == ["m1"]
        assert "m2" not in wm.user_workspaces
        assert wm.remove_member("w1", "m2", "o") is False

    def test_get_user_workspaces_skips_stale_ids(self):
        wm = WorkspaceManager({})
        ws = self.make_ws()
        wm.create_workspace(ws)
        wm.user_workspaces["o"].add("ghost")
        assert wm.get_user_workspaces("o") == [ws]
        assert wm.get_user_workspaces("nobody") == []

    def test_remove_user_workspace_unknown_user_is_noop(self):
        wm = WorkspaceManager({})
        wm._remove_user_workspace("nobody", "w1")
        assert wm.user_workspaces == {}

    def test_invalid_ids_are_reported_as_failures(self, caplog):
        wm = WorkspaceManager({})
        bad = self.make_ws(ws_id=UNHASHABLE_ID)
        assert wm.create_workspace(bad) is False
        assert wm.update_workspace(bad) is False
        assert wm.delete_workspace(UNHASHABLE_ID) is False
        assert wm.add_member(UNHASHABLE_ID, "u", "o") is False
        assert wm.remove_member(UNHASHABLE_ID, "u", "o") is False
        assert wm.workspaces == {}
        errors = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
        for fragment in (
            "Error creating workspace",
            "Error updating workspace",
            "Error deleting workspace",
            "Error adding member",
            "Error removing member",
        ):
            assert any(fragment in m for m in errors), fragment

    def test_statistics(self):
        wm = WorkspaceManager({})
        assert wm.get_workspace_statistics() == {
            "total_workspaces": 0,
            "total_members": 0,
            "average_members_per_workspace": 0,
        }
        wm.create_workspace(self.make_ws("w1", members=["a", "b"]))
        wm.create_workspace(self.make_ws("w2"))
        assert wm.get_workspace_statistics() == {
            "total_workspaces": 2,
            "total_members": 4,
            "average_members_per_workspace": 2.0,
        }


class TestActivityTracker:
    def test_default_and_configured_limit(self):
        assert ActivityTracker({}).max_activities == 10000
        tracker = ActivityTracker({"max_activities": 3})
        for i in range(5):
            tracker.log_activity(make_activity(i))
        assert [a.id for a in tracker.activities] == ["a2", "a3", "a4"]

    def test_user_and_workspace_queries_sorted_newest_first(self):
        tracker = ActivityTracker({})
        tracker.log_activity(make_activity(1, user_id="u1", workspace_id="w1"))
        tracker.log_activity(make_activity(3, user_id="u1", workspace_id="w2"))
        tracker.log_activity(make_activity(2, user_id="u2", workspace_id="w1"))

        assert [a.id for a in tracker.get_user_activities("u1")] == ["a3", "a1"]
        assert [a.id for a in tracker.get_user_activities("u1", limit=1)] == ["a3"]
        assert [a.id for a in tracker.get_workspace_activities("w1")] == ["a2", "a1"]
        assert tracker.get_workspace_activities("w1", limit=1)[0].id == "a2"
        assert tracker.get_user_activities("nobody") == []

    def test_statistics(self):
        tracker = ActivityTracker({})
        assert tracker.get_activity_statistics() == {"total_activities": 0}
        tracker.log_activity(make_activity(1, "u1", "w1", "edit"))
        tracker.log_activity(make_activity(2, "u2", "w1", "edit"))
        tracker.log_activity(make_activity(3, "u1", "w2", "view"))
        assert tracker.get_activity_statistics() == {
            "total_activities": 3,
            "action_distribution": {"edit": 2, "view": 1},
            "unique_users": 2,
            "unique_workspaces": 2,
        }


class TestCollaborationManager:
    def test_subservices_receive_their_config(self):
        cm = CollaborationManager(
            {"activity_tracker": {"max_activities": 7}, "user_management": {"k": 1}}
        )
        assert cm.activity_tracker.max_activities == 7
        assert cm.user_management.config == {"k": 1}
        assert cm.workspace_manager.config == {}

    def test_end_to_end_flow(self):
        cm = CollaborationManager({})
        owner = cm.create_user("owner", "o@x.org", UserRole.RESEARCHER)
        guest = cm.create_user("guest", "g@x.org")
        assert cm.user_management.get_user(guest).role == UserRole.STUDENT

        ws_id = cm.create_workspace("Lab", "desc", owner)
        assert cm.workspace_manager.get_workspace(ws_id).owner_id == owner

        assert cm.invite_to_workspace(ws_id, guest, owner) is True
        assert cm.invite_to_workspace(ws_id, "x", "stranger") is False
        assert [w.id for w in cm.get_user_workspaces(guest)] == [ws_id]

        # an unknown member id is ignored when resolving users
        cm.workspace_manager.get_workspace(ws_id).member_ids.append("ghost")
        members = cm.get_workspace_members(ws_id)
        assert [u.id for u in members] == [owner, guest]
        assert cm.get_workspace_members("missing") == []

        cm.log_activity(owner, ws_id, "create", "doc", "d1", {"size": 3})
        cm.log_activity(guest, ws_id, "view", "doc", "d1")
        acts = cm.activity_tracker.activities
        assert acts[0].details == {"size": 3}
        assert acts[1].details == {}

        stats = cm.get_collaboration_statistics()
        assert stats["users"]["total_users"] == 2
        assert stats["workspaces"]["total_workspaces"] == 1
        assert stats["activities"]["action_distribution"] == {"create": 1, "view": 1}

    def test_create_user_and_workspace_failures_return_none(self, monkeypatch):
        cm = CollaborationManager({})
        monkeypatch.setattr(cm.user_management, "add_user", lambda user: False)
        monkeypatch.setattr(cm.workspace_manager, "create_workspace", lambda ws: False)
        assert cm.create_user("a", "a@x.org") is None
        assert cm.create_workspace("n", "d", "o") is None
