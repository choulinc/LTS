from pathlib import Path

from lts.protocol import load_protocol, validate_protocol


ROOT = Path(__file__).resolve().parents[1]


def test_dtob6_0826_protocol_is_frozen() -> None:
    protocol = load_protocol(
        ROOT / "configs/droptest/dtob6_0826/common.yaml"
    )
    validate_protocol(protocol)
    assert protocol["protocol"]["id"] == "DTOB6-0826"
    assert protocol["protocol"]["frozen"] is True


def test_test_split_is_not_used_for_selection() -> None:
    protocol = load_protocol(
        ROOT / "configs/droptest/dtob6_0826/common.yaml"
    )
    assert protocol["checkpoint_selection"]["split"] == "validation"
    assert protocol["checkpoint_selection"]["uses_test"] is False
