"""Sprite ICQ: pixel protocol plus FakeVAE-roundtrip motion."""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from bdh_cq.video import encode_task, sample_task
from bdh_cq.video_probes import (
    STILL_L1,
    copy_detector,
    energy_centroid,
    pixel_t0_t21_l1,
    pixel_temporal_l1,
)
from bdh_cq.video_vae import FakeVideoVAE, decode_pixels


def _find_sprite(frame: np.ndarray, sprite: np.ndarray) -> tuple[int, int] | None:
    h, w = sprite.shape[:2]
    sig = sprite[:2, :2]
    win = sliding_window_view(frame, (2, 2, 3))[:, :, 0]
    matches = np.all(win == sig, axis=(2, 3, 4))
    ys, xs = np.where(matches)
    for y, x in zip(ys, xs):
        if y + h <= frame.shape[0] and x + w <= frame.shape[1]:
            if np.array_equal(frame[y : y + h, x : x + w], sprite):
                return int(x), int(y)
    return None


def test_translate_icq_pixel_query_moves_and_demos_share_direction():
    task = sample_task("translate", seed=0)
    sprite = task["params"]["sprite"]
    dx, dy = task["params"]["dx"], task["params"]["dy"]
    _, _, query = task["test"][0]
    q0 = _find_sprite(query[0], sprite)
    q1 = _find_sprite(query[-1], sprite)
    assert q0 is not None and q1 is not None
    qdx, qdy = q1[0] - q0[0], q1[1] - q0[1]
    assert abs(qdx) + abs(qdy) >= 20
    assert (qdx == 0 or np.sign(qdx) == np.sign(dx) or dx == 0)
    assert (qdy == 0 or np.sign(qdy) == np.sign(dy) or dy == 0)
    for _level, _, clip in task["train"]:
        a = _find_sprite(clip[0], sprite)
        b = _find_sprite(clip[-1], sprite)
        assert a is not None and b is not None
        ddx, ddy = b[0] - a[0], b[1] - a[1]
        if qdx:
            assert np.sign(ddx) == np.sign(qdx)
        if qdy:
            assert np.sign(ddy) == np.sign(qdy)


def test_identity_icq_pixel_query_is_still():
    task = sample_task("identity", seed=0)
    sprite = task["params"]["sprite"]
    _, _, clip = task["test"][0]
    a = _find_sprite(clip[0], sprite)
    b = _find_sprite(clip[-1], sprite)
    assert a is not None and a == b


def test_fake_vae_roundtrip_keeps_translate_motion():
    vae = FakeVideoVAE(seed=0, spatial=8)
    task = sample_task("translate", seed=0)
    sprite = task["params"]["sprite"]
    _, _, query = task["test"][0]
    q0 = _find_sprite(query[0], sprite)
    q1 = _find_sprite(query[-1], sprite)
    assert q0 is not None and q1 is not None
    lat = encode_task(vae, task)
    z = lat["query_out"]
    copy = copy_detector(z, z)
    pix = decode_pixels(vae, z)
    assert copy["mse_star_vs_still"] > 0.001
    assert pixel_temporal_l1(pix) > STILL_L1
    assert pixel_t0_t21_l1(pix) > STILL_L1
    cents = energy_centroid(z)[0]
    dlat = cents[-1] - cents[0]
    # same axis as the pixel sprite, at least one latent cell of travel
    if abs(q1[0] - q0[0]) >= abs(q1[1] - q0[1]):
        assert abs(float(dlat[1])) > abs(float(dlat[0]))
        assert abs(float(dlat[1])) >= 0.5
    else:
        assert abs(float(dlat[0])) > abs(float(dlat[1]))
        assert abs(float(dlat[0])) >= 0.5
