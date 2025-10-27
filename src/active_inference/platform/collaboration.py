"""
Platform - Collaboration Management

Multi-user collaboration features for the Active Inference Knowledge Environment.
Provides user management, workspace coordination, version control, and
collaborative editing capabilities.
"""

import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class User:
    """User in the platform"""
    id: str
    username: str
    email: str
    role: str = "user"  # user, editor, admin
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class Workspace:
    """Collaborative workspace"""
    id: str
    name: str
    description: str
    owner_id: str
    member_ids: List[str]
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class UserManagement:
    """Manages platform users"""

    def __init__(self):
        self.users: Dict[str, User] = {}
        logger.info("UserManagement initialized")

    def add_user(self, user: User) -> bool:
        """Add a new user"""
        if user.id in self.users:
            logger.warning(f"User {user.id} already exists")
            return False

        self.users[user.id] = user
        logger.info(f"Added user: {user.username}")
        return True

    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return self.users.get(user_id)

    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user (placeholder)"""
        # In real implementation, this would check credentials
        for user in self.users.values():
            if user.username == username:
                return user
        return None

    def list_users(self, role: Optional[str] = None) -> List[User]:
        """List users, optionally filtered by role"""
        users = list(self.users.values())

        if role:
            users = [user for user in users if user.role == role]

        return sorted(users, key=lambda x: x.username)


class WorkspaceManager:
    """Manages collaborative workspaces"""

    def __init__(self):
        self.workspaces: Dict[str, Workspace] = {}
        logger.info("WorkspaceManager initialized")

    def create_workspace(self, workspace: Workspace) -> bool:
        """Create a new workspace"""
        if workspace.id in self.workspaces:
            logger.warning(f"Workspace {workspace.id} already exists")
            return False

        self.workspaces[workspace.id] = workspace
        logger.info(f"Created workspace: {workspace.name}")
        return True

    def get_workspace(self, workspace_id: str) -> Optional[Workspace]:
        """Get workspace by ID"""
        return self.workspaces.get(workspace_id)

    def add_workspace_member(self, workspace_id: str, user_id: str) -> bool:
        """Add a member to a workspace"""
        workspace = self.get_workspace(workspace_id)
        if not workspace:
            return False

        if user_id not in workspace.member_ids:
            workspace.member_ids.append(user_id)
            logger.info(f"Added user {user_id} to workspace {workspace_id}")

        return True

    def remove_workspace_member(self, workspace_id: str, user_id: str) -> bool:
        """Remove a member from a workspace"""
        workspace = self.get_workspace(workspace_id)
        if not workspace:
            return False

        if user_id in workspace.member_ids:
            workspace.member_ids.remove(user_id)
            logger.info(f"Removed user {user_id} from workspace {workspace_id}")

        return True

    def list_workspaces(self, user_id: Optional[str] = None) -> List[Workspace]:
        """List workspaces, optionally filtered by user"""
        workspaces = list(self.workspaces.values())

        if user_id:
            workspaces = [ws for ws in workspaces if user_id in ws.member_ids or ws.owner_id == user_id]

        return sorted(workspaces, key=lambda x: x.name)


class CollaborationManager:
    """Main collaboration management system"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.user_management = UserManagement()
        self.workspace_manager = WorkspaceManager()

        logger.info("CollaborationManager initialized")

    def create_user(self, username: str, email: str, role: str = "user") -> Optional[str]:
        """Create a new user"""
        user_id = f"user_{username.lower().replace(' ', '_')}"

        user = User(
            id=user_id,
            username=username,
            email=email,
            role=role
        )

        if self.user_management.add_user(user):
            return user_id

        return None

    def create_workspace(self, name: str, description: str, owner_id: str) -> Optional[str]:
        """Create a new workspace"""
        workspace_id = f"workspace_{name.lower().replace(' ', '_')}"

        workspace = Workspace(
            id=workspace_id,
            name=name,
            description=description,
            owner_id=owner_id,
            member_ids=[owner_id]
        )

        if self.workspace_manager.create_workspace(workspace):
            return workspace_id

        return None

    def get_user_activity(self, user_id: str) -> Dict[str, Any]:
        """Get user activity summary"""
        user = self.user_management.get_user(user_id)
        if not user:
            return {"error": "User not found"}

        # Placeholder for actual activity tracking
        workspaces = self.workspace_manager.list_workspaces(user_id)

        return {
            "user_id": user_id,
            "username": user.username,
            "role": user.role,
            "workspaces_count": len(workspaces),
            "workspaces": [ws.name for ws in workspaces],
            "last_active": user.created_at.isoformat()
        }

    def get_collaboration_status(self) -> Dict[str, Any]:
        """Get overall collaboration status"""
        users = self.user_management.list_users()
        workspaces = self.workspace_manager.list_workspaces()

        return {
            "total_users": len(users),
            "total_workspaces": len(workspaces),
            "users_by_role": {
                role: len([u for u in users if u.role == role])
                for role in set(u.role for u in users)
            },
            "active_workspaces": len(workspaces),
            "timestamp": datetime.now().isoformat()
        }

