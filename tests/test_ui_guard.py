"""One run per profile at a time: overlapping runs queue on the single local model."""
from unittest.mock import MagicMock, patch

import streamlit_app
from streamlit_app import invoke, profile_lock


def config(thread_id):
    return {"configurable": {"thread_id": thread_id}}


def test_second_action_is_refused_while_a_run_holds_the_lock():
    graph = MagicMock()
    lock = profile_lock("guard-busy")
    assert lock.acquire(blocking=False)                  # another tab is mid-run
    try:
        assert invoke(graph, {"event": "x"}, config("guard-busy")) is False
        graph.invoke.assert_not_called()
    finally:
        lock.release()


def test_lock_is_released_after_success_and_after_failure():
    with patch.object(streamlit_app, "record_run") as record:
        ok = MagicMock()
        assert invoke(ok, {}, config("guard-free")) is True
        assert not profile_lock("guard-free").locked()

        failing = MagicMock()
        failing.invoke.side_effect = RuntimeError("model down")
        assert invoke(failing, {}, config("guard-free")) is False
        assert not profile_lock("guard-free").locked()
        assert record.call_count == 2                        # the run is still recorded


def test_profiles_do_not_block_each_other():
    lock = profile_lock("guard-a")
    assert lock.acquire(blocking=False)
    try:
        assert profile_lock("guard-a") is lock               # same profile, same lock
        assert not profile_lock("guard-b").locked()
    finally:
        lock.release()
