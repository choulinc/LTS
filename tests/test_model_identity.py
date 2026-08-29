import torch

from lts.models import LTSConfig, build_lts


def parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def test_dtob6_lts_parameter_identities() -> None:
    parts = torch.arange(17, dtype=torch.long)

    torch.manual_seed(7)
    plain = build_lts(
        LTSConfig(
            component_aware=False,
            fourier=False,
            trajectory_envelope=False,
        )
    )
    assert parameters(plain) == 8_018_444

    torch.manual_seed(7)
    ca = build_lts(
        LTSConfig(fourier=False, trajectory_envelope=False),
        node_component_index=parts,
    )
    assert parameters(ca) == 8_248_716

    torch.manual_seed(7)
    full = build_lts(LTSConfig(), node_component_index=parts)
    assert parameters(full) == 8_335_055
