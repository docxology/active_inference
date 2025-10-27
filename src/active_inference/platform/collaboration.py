"""
Collaboration Platform Service

Provides multi-user collaboration features for the Active Inference Knowledge Environment.
Includes user management, workspace management, and collaborative content creation.
"""

import logging
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class UserRole(Enum):
    """User roles for access control"""
    ADMIN = "admin"
    RESEARCHER = "researcher"
    EDUCATOR = "educator"
    STUDENT = "student"
    GUEST = "guest"


class Permission(Enum):
    """Permissions for collaborative actions"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    SHARE = "share"
    ADMIN = "admin"


@dataclass
class User:
    """User representation"""
    id: str
    username: str
    email: str
    role: UserRole
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    profile: Dict[str, Any] = field(default_factory=dict)

    def has_permission(self, permission: Permission) -> bool:
        """Check if user has specific permission"""
        role_permissions = {
            UserRole.ADMIN: {Permission.READ, Permission.WRITE, Permission.DELETE, Permission.SHARE, Permission.ADMIN},
            UserRole.RESEARCHER: {Permission.READ, Permission.WRITE, Permission.SHARE},
            UserRole.EDUCATOR: {Permission.READ, Permission.WRITE, Permission.SHARE},
            UserRole.STUDENT: {Permission.READ},
            UserRole.GUEST: {Permission.READ}
        }

        return permission in role_permissions.get(self.role, set())


@dataclass
class Workspace:
    """Workspace for collaborative work"""
    id: str
    name: str
    description: str
    owner_id: str
    member_ids: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    settings: Dict[str, Any] = field(default_factory=dict)

    def add_member(self, user_id: str) -> bool:
        """Add member to workspace"""
        if user_id not in self.member_ids:
            self.member_ids.append(user_id)
            self.last_modified = datetime.now()
            return True
        return False

    def remove_member(self, user_id: str) -> bool:
        """Remove member from workspace"""
        if user_id in self.member_ids:
            self.member_ids.remove(user_id)
            self.last_modified = datetime.now()
            return True
        return False

    def is_member(self, user_id: str) -> bool:
        """Check if user is a member"""
        return user_id == self.owner_id or user_id in self.member_ids


@dataclass
class Activity:
    """Activity log entry"""
    id: str
    user_id: str
    workspace_id: str
    action: str
    resource_type: str
    resource_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)


class UserManagement:
    """Manages user accounts and authentication"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize user management"""
        self.config = config
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Dict[str, Any]] = {}  # session_id -> user info

        logger.info("User management initialized")

    def add_user(self, user: User) -> bool:
        """Add new user"""
        try:
            if user.id in self.users:
                logger.warning(f"User {user.id} already exists")
                return False

            self.users[user.id] = user
            logger.info(f"Added user: {user.username} ({user.id})")
            return True

        except Exception as e:
            logger.error(f"Error adding user {user.id}: {e}")
            return False

    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return self.users.get(user_id)

    def update_user(self, user: User) -> bool:
        """Update user information"""
        try:
            if user.id not in self.users:
                return False

            self.users[user.id] = user
            logger.info(f"Updated user: {user.username} ({user.id})")
            return True

        except Exception as e:
            logger.error(f"Error updating user {user.id}: {e}")
            return False

    def delete_user(self, user_id: str) -> bool:
        """Delete user"""
        try:
            if user_id not in self.users:
                return False

            del self.users[user_id]
            logger.info(f"Deleted user: {user_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting user {user_id}: {e}")
            return False

    def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """Authenticate user and return session ID"""
        # Simplified authentication (in real implementation, this would verify password)
        for user in self.users.values():
            if user.username == username:
                # In a real system, verify password hash
                session_id = str(uuid.uuid4())
                self.sessions[session_id] = {
                    'user_id': user.id,
                    'username': user.username,
                    'role': user.role.value,
                    'created_at': datetime.now()
                }
                user.last_active = datetime.now()
                return session_id

        return None

    def validate_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Validate session and return user info"""
        return self.sessions.get(session_id)

    def logout_user(self, session_id: str) -> bool:
        """End user session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False

    def get_user_statistics(self) -> Dict[str, Any]:
        """Get user statistics"""
        roles = {}
        for user in self.users.values():
            role = user.role.value
            roles[role] = roles.get(role, 0) + 1

        return {
            'total_users': len(self.users),
            'active_sessions': len(self.sessions),
            'role_distribution': roles
        }


class WorkspaceManager:
    """Manages collaborative workspaces"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize workspace manager"""
        self.config = config
        self.workspaces: Dict[str, Workspace] = {}
        self.user_workspaces: Dict[str, Set[str]] = {}  # user_id -> set of workspace_ids

        logger.info("Workspace manager initialized")

    def create_workspace(self, workspace: Workspace) -> bool:
        """Create new workspace"""
        try:
            if workspace.id in self.workspaces:
                logger.warning(f"Workspace {workspace.id} already exists")
                return False

            self.workspaces[workspace.id] = workspace

            # Update user-workspace mapping
            self._add_user_workspace(workspace.owner_id, workspace.id)
            for member_id in workspace.member_ids:
                self._add_user_workspace(member_id, workspace.id)

            logger.info(f"Created workspace: {workspace.name} ({workspace.id})")
            return True

        except Exception as e:
            logger.error(f"Error creating workspace {workspace.id}: {e}")
            return False

    def get_workspace(self, workspace_id: str) -> Optional[Workspace]:
        """Get workspace by ID"""
        return self.workspaces.get(workspace_id)

    def update_workspace(self, workspace: Workspace) -> bool:
        """Update workspace information"""
        try:
            if workspace.id not in self.workspaces:
                return False

            old_workspace = self.workspaces[workspace.id]
            self.workspaces[workspace.id] = workspace

            # Update user-workspace mappings if membership changed
            old_members = set([old_workspace.owner_id] + old_workspace.member_ids)
            new_members = set([workspace.owner_id] + workspace.member_ids)

            # Remove old members
            for user_id in old_members - new_members:
                self._remove_user_workspace(user_id, workspace.id)

            # Add new members
            for user_id in new_members - old_members:
                self._add_user_workspace(user_id, workspace.id)

            logger.info(f"Updated workspace: {workspace.name} ({workspace.id})")
            return True

        except Exception as e:
            logger.error(f"Error updating workspace {workspace.id}: {e}")
            return False

    def delete_workspace(self, workspace_id: str) -> bool:
        """Delete workspace"""
        try:
            if workspace_id not in self.workspaces:
                return False

            workspace = self.workspaces[workspace_id]

            # Remove from user-workspace mappings
            for user_id in [workspace.owner_id] + workspace.member_ids:
                self._remove_user_workspace(user_id, workspace_id)

            del self.workspaces[workspace_id]
            logger.info(f"Deleted workspace: {workspace_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting workspace {workspace_id}: {e}")
            return False

    def add_member(self, workspace_id: str, user_id: str, requester_id: str) -> bool:
        """Add member to workspace"""
        try:
            workspace = self.workspaces.get(workspace_id)
            if not workspace:
                return False

            # Check if requester has permission
            if requester_id != workspace.owner_id and requester_id not in workspace.member_ids:
                return False

            if workspace.add_member(user_id):
                self._add_user_workspace(user_id, workspace_id)
                logger.info(f"Added member {user_id} to workspace {workspace_id}")
                return True

            return False

        except Exception as e:
            logger.error(f"Error adding member to workspace {workspace_id}: {e}")
            return False

    def remove_member(self, workspace_id: str, user_id: str, requester_id: str) -> bool:
        """Remove member from workspace"""
        try:
            workspace = self.workspaces.get(workspace_id)
            if not workspace:
                return False

            # Only owner can remove members
            if requester_id != workspace.owner_id:
                return False

            if workspace.remove_member(user_id):
                self._remove_user_workspace(user_id, workspace_id)
                logger.info(f"Removed member {user_id} from workspace {workspace_id}")
                return True

            return False

        except Exception as e:
            logger.error(f"Error removing member from workspace {workspace_id}: {e}")
            return False

    def get_user_workspaces(self, user_id: str) -> List[Workspace]:
        """Get all workspaces for a user"""
        workspace_ids = self.user_workspaces.get(user_id, set())
        return [self.workspaces[wid] for wid in workspace_ids if wid in self.workspaces]

    def _add_user_workspace(self, user_id: str, workspace_id: str):
        """Add workspace to user's workspace list"""
        if user_id not in self.user_workspaces:
            self.user_workspaces[user_id] = set()
        self.user_workspaces[user_id].add(workspace_id)

    def _remove_user_workspace(self, user_id: str, workspace_id: str):
        """Remove workspace from user's workspace list"""
        if user_id in self.user_workspaces:
            self.user_workspaces[user_id].discard(workspace_id)
            if not self.user_workspaces[user_id]:
                del self.user_workspaces[user_id]

    def get_workspace_statistics(self) -> Dict[str, Any]:
        """Get workspace statistics"""
        total_members = sum(len(ws.member_ids) + 1 for ws in self.workspaces.values())  # +1 for owner

        return {
            'total_workspaces': len(self.workspaces),
            'total_members': total_members,
            'average_members_per_workspace': total_members / len(self.workspaces) if self.workspaces else 0
        }


class ActivityTracker:
    """Tracks user activities for analytics"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize activity tracker"""
        self.config = config
        self.activities: List[Activity] = []
        self.max_activities = config.get('max_activities', 10000)

    def log_activity(self, activity: Activity):
        """Log user activity"""
        self.activities.append(activity)

        # Maintain size limit
        if len(self.activities) > self.max_activities:
            self.activities = self.activities[-self.max_activities:]

        logger.debug(f"Logged activity: {activity.action} by {activity.user_id}")

    def get_user_activities(self, user_id: str, limit: int = 50) -> List[Activity]:
        """Get activities for a user"""
        user_activities = [a for a in self.activities if a.user_id == user_id]
        return sorted(user_activities, key=lambda x: x.timestamp, reverse=True)[:limit]

    def get_workspace_activities(self, workspace_id: str, limit: int = 50) -> List[Activity]:
        """Get activities for a workspace"""
        workspace_activities = [a for a in self.activities if a.workspace_id == workspace_id]
        return sorted(workspace_activities, key=lambda x: x.timestamp, reverse=True)[:limit]

    def get_activity_statistics(self) -> Dict[str, Any]:
        """Get activity statistics"""
        if not self.activities:
            return {'total_activities': 0}

        actions = {}
        for activity in self.activities:
            actions[activity.action] = actions.get(activity.action, 0) + 1

        return {
            'total_activities': len(self.activities),
            'action_distribution': actions,
            'unique_users': len(set(a.user_id for a in self.activities)),
            'unique_workspaces': len(set(a.workspace_id for a in self.activities))
        }


class CollaborationManager:
    """Main collaboration manager coordinating all collaboration services"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize collaboration manager"""
        self.config = config

        # Initialize services
        self.user_management = UserManagement(config.get('user_management', {}))
        self.workspace_manager = WorkspaceManager(config.get('workspace_manager', {}))
        self.activity_tracker = ActivityTracker(config.get('activity_tracker', {}))

        logger.info("Collaboration manager initialized")

    def create_user(self, username: str, email: str, role: UserRole = UserRole.STUDENT) -> Optional[str]:
        """Create new user"""
        user_id = str(uuid.uuid4())
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
        """Create new workspace"""
        workspace_id = str(uuid.uuid4())
        workspace = Workspace(
            id=workspace_id,
            name=name,
            description=description,
            owner_id=owner_id
        )

        if self.workspace_manager.create_workspace(workspace):
            return workspace_id
        return None

    def invite_to_workspace(self, workspace_id: str, user_id: str, inviter_id: str) -> bool:
        """Invite user to workspace"""
        return self.workspace_manager.add_member(workspace_id, user_id, inviter_id)

    def log_activity(self, user_id: str, workspace_id: str, action: str,
                    resource_type: str, resource_id: str, details: Dict[str, Any] = None):
        """Log user activity"""
        activity = Activity(
            id=str(uuid.uuid4()),
            user_id=user_id,
            workspace_id=workspace_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details or {}
        )
        self.activity_tracker.log_activity(activity)

    def get_user_workspaces(self, user_id: str) -> List[Workspace]:
        """Get workspaces for user"""
        return self.workspace_manager.get_user_workspaces(user_id)

    def get_workspace_members(self, workspace_id: str) -> List[User]:
        """Get members of workspace"""
        workspace = self.workspace_manager.get_workspace(workspace_id)
        if not workspace:
            return []

        members = []
        for user_id in [workspace.owner_id] + workspace.member_ids:
            user = self.user_management.get_user(user_id)
            if user:
                members.append(user)

        return members

    def get_collaboration_statistics(self) -> Dict[str, Any]:
        """Get comprehensive collaboration statistics"""
        return {
            'users': self.user_management.get_user_statistics(),
            'workspaces': self.workspace_manager.get_workspace_statistics(),
            'activities': self.activity_tracker.get_activity_statistics()
        }