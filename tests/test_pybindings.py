import numpy as np

def test_interp():
    from spano import Mapping

    mid = Mapping.interpolate_scalar(
        [0, 1], [Mapping.identity(), Mapping.shift(10, 20)], 0.5
    )
    assert np.allclose(mid.mat, Mapping.shift(5, 10).mat)


def test_warp_array():
    try:
        import imageio.v3 as iio
    except ImportError:
        import imageio as iio
    from spano import Mapping

    src = np.array(iio.imread("tests/source.png")).astype(np.float32)
    dst = np.array(iio.imread("tests/target.png")).astype(np.float32)

    map = (
        Mapping.from_matrix(
            np.array(
                [
                    [0.47654548, -0.045553986, 4.847797],
                    [-0.14852144, 0.6426208, 2.1364543],
                    [-0.009891294, -0.0021317923, 0.88151735],
                ],
                dtype=np.float32,
            ),
            kind="Projective",
        )
        .inverse()
        .rescale(1 / 16)
    )

    warped, _ = map.warp_array(src, (480, 640), [128, 0, 0])
    assert np.allclose(warped, dst, atol=1)


def test_transform_types():
    from spano import TransformationType

    for var in TransformationType.variants():
        assert var == TransformationType.from_str(var.to_str())
        assert var == TransformationType.from_str(var.to_str().upper())
        assert var == TransformationType.from_str(var.to_str().lower())
