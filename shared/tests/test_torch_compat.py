"""Tests for the platform-guarded torch workaround."""

import platform
import sys

import pytest

from shared.lib.torch_compat import apply_torch_workarounds

torch = pytest.importorskip("torch")


@pytest.fixture
def mkldnn_state():
    """Restore the global mkldnn flag after each test."""
    original = torch.backends.mkldnn.enabled
    yield
    torch.backends.mkldnn.enabled = original


def test_disables_mkldnn_on_linux_arm64(monkeypatch, mkldnn_state):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(platform, "machine", lambda: "aarch64")
    torch.backends.mkldnn.enabled = True

    assert apply_torch_workarounds() is True
    assert torch.backends.mkldnn.enabled is False


@pytest.mark.parametrize(
    ("plat", "machine"),
    [("linux", "x86_64"), ("darwin", "arm64"), ("win32", "AMD64")],
)
def test_leaves_other_platforms_untouched(monkeypatch, mkldnn_state, plat, machine):
    monkeypatch.setattr(sys, "platform", plat)
    monkeypatch.setattr(platform, "machine", lambda: machine)
    before = torch.backends.mkldnn.enabled

    assert apply_torch_workarounds() is False
    assert torch.backends.mkldnn.enabled is before
