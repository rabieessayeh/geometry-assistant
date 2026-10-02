"""The per-session store of uploaded models."""

from app.sessions import SessionStore

NO_RESERVED = frozenset()


def test_unknown_session_has_no_upload(meshes):
    store = SessionStore(max_sessions=2, max_uploads=2)
    assert store.catalog(None).models == {}
    assert store.catalog("nobody").models == {}
    assert store._sessions == {}  # looking up a session does not create it


def test_sessions_are_isolated(meshes):
    store = SessionStore(max_sessions=2, max_uploads=2)
    assert store.add("alice", "part", meshes["box"], NO_RESERVED) == "part"
    assert store.add("bob", "part", meshes["wedge"], NO_RESERVED) == "part"
    assert store.catalog("alice").models["part"] is meshes["box"]
    assert store.catalog("bob").models["part"] is meshes["wedge"]


def test_ids_avoid_reserved_names(meshes):
    store = SessionStore(max_sessions=2, max_uploads=2)
    assert store.add("alice", "bridge", meshes["box"], frozenset({"bridge"})) == "bridge_2"


def test_oldest_upload_is_dropped(meshes):
    store = SessionStore(max_sessions=2, max_uploads=2)
    for name in ("a", "b", "c"):
        store.add("alice", name, meshes["box"], NO_RESERVED)
    assert list(store.catalog("alice").models) == ["b", "c"]


def test_least_recently_used_session_is_dropped(meshes):
    store = SessionStore(max_sessions=2, max_uploads=2)
    store.add("alice", "a", meshes["box"], NO_RESERVED)
    store.add("bob", "b", meshes["box"], NO_RESERVED)
    store.catalog("alice")  # alice is now more recent than bob
    store.add("carol", "c", meshes["box"], NO_RESERVED)
    assert store.catalog("bob").models == {}
    assert list(store.catalog("alice").models) == ["a"]
    assert list(store.catalog("carol").models) == ["c"]
