import pytest
import torch

from isp_ai_enhancement.data.context import (
    ContextBuilder,
    ContextConfig,
    RawMetadata,
    canonical_pack_bayer,
    load_context_config,
)


def test_canonical_pack_rggb() -> None:
    raw = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    packed = canonical_pack_bayer(raw, "RGGB")
    assert packed.shape == (4, 2, 2)
    assert packed[0, 0, 0].item() == 0
    assert packed[1, 0, 0].item() == 1
    assert packed[2, 0, 0].item() == 4
    assert packed[3, 0, 0].item() == 5


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        ("RGGB", [0, 1, 4, 5]),
        ("GRBG", [1, 0, 5, 4]),
        ("GBRG", [4, 5, 0, 1]),
        ("BGGR", [5, 4, 1, 0]),
    ],
)
def test_all_cfa_patterns(pattern: str, expected: list[int]) -> None:
    raw = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    packed = canonical_pack_bayer(raw, pattern)
    assert packed[:, 0, 0].tolist() == expected


def test_context_contract() -> None:
    builder = ContextBuilder(ContextConfig(camera_embeddings={"sensor": (0.1, -0.1, 0.2, -0.2)}))
    raw = torch.full((1, 4, 32, 48), 0.25)
    result = builder.build(
        raw,
        RawMetadata(
            sensor_id="sensor",
            noise_sigma=0.2,
            exposure_ratio=1.0,
            wb_rg=2.0,
            wb_bg=1.5,
        ),
    )
    assert result.shape == (1, 16, 32, 48)
    assert torch.allclose(result[:, 5], torch.tensor(0.5))
    assert torch.all(result[:, 6] == 1)
    assert torch.all(result[:, 7] == 0)
    assert torch.all(result[:, 15] == 1)


def test_context_rejects_unknown_sensor() -> None:
    with pytest.raises(ValueError, match="missing camera embedding"):
        ContextBuilder().build(torch.zeros(1, 4, 16, 16), RawMetadata(sensor_id="unknown"))


def test_load_context_config(tmp_path) -> None:
    path = tmp_path / "context.yaml"
    path.write_text(
        "context:\n"
        "  camera_embeddings:\n"
        "    target_sensor: [0.1, -0.1, 0.2, -0.2]\n"
        "  mode_codes: {single: 0.0, hdr: 0.5, mfnr: 1.0}\n",
        encoding="utf-8",
    )
    config = load_context_config(path)
    assert config.camera_embeddings["target_sensor"] == (0.1, -0.1, 0.2, -0.2)


def test_context_config_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="unknown context"):
        ContextConfig.from_mapping({"camera_embeddings": {}, "typo": 1})
