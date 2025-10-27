# Platform Collaboration - Agent Development Guide

**Guidelines for AI agents working with platform collaboration features.**

## 🤖 Agent Role & Responsibilities

**What agents should do when working with collaboration features:**

### Primary Responsibilities
- **Feature Development**: Implement collaborative editing and communication features
- **Real-time Systems**: Build WebSocket-based real-time collaboration
- **User Management**: Develop authentication and authorization systems
- **State Synchronization**: Ensure consistent state across all users
- **Conflict Resolution**: Implement merge and conflict resolution algorithms

### Development Focus Areas
1. **Real-time Collaboration**: WebSocket communication and live updates
2. **User Experience**: Intuitive collaborative interfaces
3. **Data Consistency**: Maintain data integrity across sessions
4. **Performance**: Optimize for multiple concurrent users
5. **Security**: Secure multi-user access and permissions

## 🏗️ Architecture & Integration

### Collaboration Architecture

**Understanding the collaboration system structure:**

```
Platform Services
├── Authentication & Authorization
├── Real-time Communication (WebSocket)
├── Session Management
├── Conflict Resolution
└── Activity Monitoring
```

### Integration Points

**Key integration points for collaboration:**

#### Platform Integration
- **User Service**: Authentication and user management
- **Knowledge Service**: Collaborative content editing
- **Search Service**: Index collaborative content
- **Notification Service**: User activity notifications

#### External Systems
- **WebSocket Servers**: Real-time communication
- **Database**: Session and user state persistence
- **Cache**: Real-time state caching
- **Monitoring**: User activity tracking

### Data Flow Patterns

```python
# Real-time collaboration flow
user_action → validate_permission → broadcast_to_session → update_state → notify_users
```

## 💻 Development Patterns

### Required Implementation Patterns

**All collaboration development must follow these patterns:**

#### 1. Real-time Communication Pattern
```python
class CollaborationManager:
    """Manage real-time collaborative sessions"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.websocket_manager = WebSocketManager(config)
        self.session_manager = SessionManager(config)
        self.state_manager = StateManager(config)

    async def create_session(self, content_id: str, user_id: str) -> str:
        """Create new collaborative session"""
        # Validate permissions
        await self.validate_permission(user_id, "create_session")

        # Create session
        session = await self.session_manager.create_session(content_id, user_id)

        # Set up real-time communication
        await self.websocket_manager.setup_session(session.id)

        return session.id

    async def join_session(self, session_id: str, user_id: str) -> bool:
        """Join existing collaborative session"""
        # Validate session exists
        session = await self.session_manager.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        # Validate permissions
        await self.validate_permission(user_id, "join_session")

        # Add user to session
        await self.session_manager.add_user(session_id, user_id)

        # Set up WebSocket connection
        await self.websocket_manager.add_user(session_id, user_id)

        return True
```

#### 2. State Synchronization Pattern
```python
class StateSynchronizer:
    """Synchronize state across collaborative session"""

    async def broadcast_change(self, session_id: str, change: Dict[str, Any]) -> None:
        """Broadcast change to all session participants"""
        # Validate change
        validated_change = await self.validate_change(change)

        # Update local state
        await self.state_manager.update_state(session_id, validated_change)

        # Broadcast to all users
        await self.websocket_manager.broadcast(session_id, {
            "type": "state_update",
            "change": validated_change,
            "timestamp": datetime.utcnow()
        })

    async def handle_conflict(self, session_id: str, conflict: Conflict) -> Resolution:
        """Handle conflicting changes from multiple users"""
        # Analyze conflict
        conflict_analysis = await self.analyze_conflict(conflict)

        # Attempt automatic resolution
        if conflict_analysis["auto_resolvable"]:
            return await self.auto_resolve(conflict)

        # Escalate to manual resolution
        return await self.manual_resolution(conflict)
```

#### 3. Permission Management Pattern
```python
class PermissionManager:
    """Manage user permissions for collaboration"""

    async def validate_permission(self, user_id: str, action: str) -> bool:
        """Validate user has permission for action"""
        user_permissions = await self.get_user_permissions(user_id)
        required_permission = PERMISSION_MAP[action]

        if required_permission not in user_permissions:
            await self.log_unauthorized_access(user_id, action)
            return False

        return True

    async def grant_permission(self, user_id: str, permission: str) -> bool:
        """Grant specific permission to user"""
        # Validate admin permission
        if not await self.validate_permission(current_user, "grant_permission"):
            raise PermissionError("Insufficient permissions")

        # Grant permission
        await self.user_service.add_permission(user_id, permission)

        # Log permission change
        await self.audit_log_permission_change(user_id, permission, "grant")

        return True
```

## 🧪 Testing Standards

### Test Categories (MANDATORY)

#### 1. Real-time Testing
```python
class TestRealTimeCollaboration:
    """Test real-time collaboration features"""

    async def test_concurrent_editing(self):
        """Test multiple users editing simultaneously"""
        # Create test session
        session_id = await self.create_test_session()

        # Simulate multiple users
        users = await self.create_test_users(5)

        # Concurrent edits
        tasks = [
            self.simulate_user_edit(user, session_id, edit_data)
            for user, edit_data in zip(users, edit_batches)
        ]

        # Execute concurrent edits
        results = await asyncio.gather(*tasks)

        # Validate all edits applied
        assert all(result["success"] for result in results)

        # Validate final state consistency
        final_state = await self.get_session_state(session_id)
        assert self.validate_state_consistency(final_state)

    async def test_conflict_resolution(self):
        """Test automatic conflict resolution"""
        # Create conflicting edits
        conflict_scenario = await self.create_conflict_scenario()

        # Apply conflict resolution
        resolution = await self.collaboration_manager.resolve_conflict(conflict_scenario)

        # Validate resolution
        assert resolution["resolved"] == True
        assert resolution["auto_resolved"] == True
```

#### 2. Permission Testing
```python
class TestPermissionManagement:
    """Test permission and access control"""

    async def test_permission_validation(self):
        """Test permission validation logic"""
        # Test valid permission
        user = await self.create_test_user(permissions=["read", "write"])
        assert await self.validate_permission(user.id, "write") == True

        # Test invalid permission
        user_no_write = await self.create_test_user(permissions=["read"])
        assert await self.validate_permission(user_no_write.id, "write") == False

    async def test_role_based_access(self):
        """Test role-based access control"""
        # Test different user roles
        roles = ["viewer", "editor", "admin"]

        for role in roles:
            user = await self.create_test_user(role=role)
            permissions = await self.get_role_permissions(role)

            for permission in permissions:
                assert await self.validate_permission(user.id, permission) == True
```

#### 3. State Consistency Testing
```python
class TestStateConsistency:
    """Test state consistency across sessions"""

    async def test_state_synchronization(self):
        """Test state synchronization across users"""
        # Create multi-user session
        session = await self.create_multi_user_session(user_count=10)

        # Make changes from different users
        changes = await self.generate_concurrent_changes(10)

        # Apply changes
        await self.apply_concurrent_changes(session.id, changes)

        # Validate all users see same state
        user_states = await self.get_all_user_states(session.id)
        assert all(self.compare_states(state1, state2) for state1, state2 in combinations(user_states, 2))
```

### Test Coverage Requirements

- **Real-time Features**: 100% coverage of WebSocket and state sync
- **Permission System**: 100% coverage of access control
- **Error Handling**: 100% coverage of error scenarios
- **Performance**: Concurrent user scenarios

## 📖 Documentation Standards

### Documentation Requirements (MANDATORY)

#### 1. API Documentation
**All collaboration APIs must be documented:**

```python
def create_collaborative_session(content_id: str, user_id: str) -> str:
    """
    Create a new collaborative editing session.

    This function creates a collaborative session for multiple users to edit
    the same content simultaneously with real-time synchronization.

    Args:
        content_id: Unique identifier for the content to collaborate on
        user_id: ID of the user creating the session (must have write permissions)

    Returns:
        Session ID string for joining the collaborative session

    Raises:
        PermissionError: If user lacks permission to create sessions
        ContentNotFoundError: If content_id doesn't exist
        SessionLimitError: If maximum concurrent sessions exceeded

    Examples:
        >>> session_id = create_collaborative_session("entropy_basics", "user123")
        >>> print(f"Created session: {session_id}")
        "session_abc123"
    """
    pass
```

#### 2. Configuration Documentation
**All configuration options must be documented:**

```python
# Collaboration configuration schema
collaboration_config_schema = {
    "websocket_settings": {
        "url": "ws://localhost:8080",
        "max_connections": 1000,
        "heartbeat_interval": 30
    },
    "session_settings": {
        "max_duration": 3600,  # seconds
        "max_users": 50,
        "auto_save_interval": 30
    },
    "permission_settings": {
        "default_permissions": ["read", "comment"],
        "admin_permissions": ["read", "write", "admin", "manage_users"]
    }
}
```

#### 3. User Guide Documentation
**Usage patterns must be documented:**

```markdown
## Real-time Collaboration Usage

### Creating a Session
```python
# Initialize collaboration service
collab = CollaborationService(config)

# Create collaborative session
session_id = await collab.create_session("knowledge_node_id", "user_id")

# Join the session
await collab.join_session(session_id, "another_user_id")
```

### Handling Conflicts
```python
# Automatic conflict resolution
if conflict_detected:
    resolution = await collab.resolve_conflict(conflict)
    if resolution.auto_resolved:
        # Apply automatic resolution
        await collab.apply_resolution(resolution)
    else:
        # Manual resolution required
        await collab.escalate_to_manual_resolution(conflict)
```

### Real-time Updates
```python
# WebSocket message handling
@websocket_handler
async def handle_collaboration_message(message):
    if message.type == "content_change":
        await update_content(message.data)
    elif message.type == "user_joined":
        await notify_session_users(message.user_id + " joined")
    elif message.type == "conflict":
        await handle_conflict_message(message.conflict)
```

## 🚀 Performance Optimization

### Performance Requirements

**Collaboration system must meet these performance standards:**

- **Response Time**: <50ms for real-time updates
- **Concurrent Users**: Support 1000+ simultaneous users per session
- **State Sync**: <100ms state synchronization lag
- **Memory Usage**: Efficient memory usage for large sessions

### Optimization Techniques

#### 1. Connection Optimization
```python
class WebSocketOptimizer:
    """Optimize WebSocket connections"""

    def optimize_connections(self, connections: List[WebSocketConnection]) -> None:
        """Optimize connection management"""
        # Connection pooling
        self.connection_pool = ConnectionPool(max_connections=1000)

        # Heartbeat optimization
        self.heartbeat_manager = OptimizedHeartbeat(
            interval=30,
            timeout=10
        )

        # Message batching
        self.message_batcher = MessageBatcher(
            batch_size=10,
            flush_interval=50  # ms
        )
```

#### 2. State Management Optimization
```python
class StateOptimizer:
    """Optimize state management"""

    def optimize_state_sync(self, session: CollaborationSession) -> None:
        """Optimize state synchronization"""
        # Differential updates
        self.differential_updater = DifferentialStateUpdater()

        # State compression
        self.state_compressor = StateCompressor(
            algorithm="delta_encoding"
        )

        # Conflict prediction
        self.conflict_predictor = ConflictPredictor(
            model="ml_based"
        )
```

#### 3. Resource Management
```python
class ResourceManager:
    """Manage collaboration resources efficiently"""

    def manage_session_resources(self, session: CollaborationSession) -> None:
        """Manage resources for session"""
        # Memory management
        self.memory_manager = SessionMemoryManager(
            max_memory_per_session="100MB"
        )

        # Connection management
        self.connection_manager = ConnectionManager(
            max_connections_per_session=50
        )

        # Cleanup management
        self.cleanup_manager = SessionCleanupManager(
            cleanup_interval=300  # 5 minutes
        )
```

## 🔒 Security Standards

### Security Requirements (MANDATORY)

#### 1. Authentication & Authorization
```python
class CollaborationSecurity:
    """Security for collaboration features"""

    async def authenticate_user(self, credentials: Dict) -> User:
        """Authenticate user for collaboration"""
        # Multi-factor authentication
        user = await self.auth_service.authenticate(credentials)

        # Session validation
        await self.validate_session_security(user)

        return user

    async def authorize_action(self, user: User, action: str, resource: str) -> bool:
        """Authorize user action on resource"""
        # Permission checking
        has_permission = await self.permission_service.check_permission(
            user.id, action, resource
        )

        # Security logging
        await self.security_logger.log_access(user.id, action, resource)

        return has_permission
```

#### 2. Data Protection
```python
class DataProtection:
    """Protect collaborative data"""

    def encrypt_session_data(self, data: Dict) -> str:
        """Encrypt session data for transmission"""
        return self.encryption_service.encrypt(data)

    def validate_data_integrity(self, data: Dict) -> bool:
        """Validate data integrity"""
        return self.integrity_checker.validate(data)

    def sanitize_collaborative_input(self, input_data: Dict) -> Dict:
        """Sanitize user input for collaboration"""
        return self.sanitizer.sanitize(input_data)
```

## 🐛 Debugging & Troubleshooting

### Debug Configuration

```python
# Enable collaboration debugging
debug_config = {
    "debug_mode": True,
    "log_level": "DEBUG",
    "websocket_debug": True,
    "state_debug": True,
    "performance_debug": True
}

collab_manager = CollaborationManager(debug_config)
```

### Common Debugging Patterns

#### 1. Real-time Debugging
```python
class CollaborationDebugger:
    """Debug collaboration issues"""

    async def debug_session_state(self, session_id: str) -> Dict:
        """Debug session state consistency"""
        session = await self.get_session(session_id)

        debug_info = {
            "users": await self.get_session_users(session_id),
            "state": await self.get_session_state(session_id),
            "websocket_connections": await self.get_websocket_status(session_id),
            "recent_changes": await self.get_recent_changes(session_id),
            "conflict_history": await self.get_conflict_history(session_id)
        }

        return debug_info

    async def trace_message_flow(self, message_id: str) -> List[Message]:
        """Trace message flow through system"""
        return await self.message_tracer.trace_message(message_id)
```

#### 2. Performance Profiling
```python
class PerformanceProfiler:
    """Profile collaboration performance"""

    async def profile_real_time_updates(self, session_id: str) -> Dict:
        """Profile real-time update performance"""
        profiler = RealTimeProfiler()

        with profiler.monitor_session(session_id):
            # Simulate real-time activity
            await self.simulate_concurrent_activity(session_id)

        return profiler.get_metrics()

    async def profile_state_synchronization(self, session_id: str) -> Dict:
        """Profile state synchronization performance"""
        sync_profiler = StateSyncProfiler()

        await sync_profiler.start_monitoring(session_id)

        # Generate state changes
        await self.generate_state_changes(session_id, count=1000)

        await sync_profiler.stop_monitoring()

        return sync_profiler.get_report()
```

## 🔄 Development Workflow

### Agent Development Process

1. **Task Assessment**
   - Understand collaboration requirements
   - Analyze real-time constraints
   - Consider user experience implications

2. **Architecture Planning**
   - Design WebSocket communication
   - Plan state management strategy
   - Consider scalability requirements

3. **Test-Driven Development**
   - Write tests for real-time features first
   - Test concurrent user scenarios
   - Validate state consistency

4. **Implementation**
   - Implement WebSocket handlers
   - Add state synchronization
   - Implement permission system

5. **Quality Assurance**
   - Test with multiple concurrent users
   - Validate real-time performance
   - Ensure security compliance

6. **Integration**
   - Test with knowledge platform
   - Validate user experience
   - Performance optimization

### Code Review Checklist

**Before submitting collaboration code for review:**

- [ ] **Real-time Tests**: Comprehensive WebSocket and state sync tests
- [ ] **Concurrent Users**: Tested with multiple simultaneous users
- [ ] **State Consistency**: Validated state synchronization
- [ ] **Permission System**: Complete access control implementation
- [ ] **Error Handling**: Robust error handling for real-time scenarios
- [ ] **Performance**: Meets response time requirements
- [ ] **Security**: Secure authentication and authorization
- [ ] **Documentation**: Complete API and usage documentation

## 📚 Learning Resources

### Collaboration-Specific Resources

- **[WebSocket Best Practices](https://example.com/websockets)**: Real-time communication
- **[State Management Patterns](https://example.com/state-management)**: State synchronization
- **[Conflict Resolution](https://example.com/conflict-resolution)**: Merge algorithms
- **[Real-time Systems](https://example.com/realtime-systems)**: System design

### Platform Integration

- **[Platform Services](../../platform/README.md)**: Platform architecture
- **[Knowledge Integration](../../../knowledge/README.md)**: Knowledge platform
- **[User Management](../../../src/active_inference/platform/README.md)**: Authentication

## 🎯 Success Metrics

### Quality Metrics

- **Real-time Performance**: <50ms response time for updates
- **Concurrent Users**: Support 1000+ simultaneous users
- **State Consistency**: 100% state synchronization accuracy
- **User Experience**: Intuitive collaborative interface

### Development Metrics

- **Feature Completeness**: All planned collaboration features implemented
- **Performance**: Meets real-time performance requirements
- **Security**: Zero security vulnerabilities in collaboration
- **Integration**: Seamless integration with platform services

---

**Component**: Platform Collaboration | **Version**: 1.0.0 | **Last Updated**: October 2024

*"Active Inference for, with, by Generative AI"* - Collaborative intelligence through real-time knowledge creation.
