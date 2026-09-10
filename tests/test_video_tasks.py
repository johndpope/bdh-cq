"""oracle invariants for the v0 video families."""

from numpy.lib.stride_tricks import sliding_window_view

import numpy as np
import pytest

from bdh_cq.video_tasks import (
    ANCHOR, ANCHOR_GRAY, BAR_H, CANVAS, FRAMES, SEED_W, SPRITE_SIZE, STAMP_SIZE,
    VIDEO_TASKS, Identity, Pan, Recolor, SeedExtend, StampCopy, Translate,
    TranslatePan, blit_clip, task_at_level, trajectory
)


def find_sprite(frame: np.ndarray, sprite: np.ndarray) -> tuple[int, int]:
    h, w = sprite.shape[:2]
    sig = sprite[:2, :2]
    win = sliding_window_view(frame, (2, 2, 3))[:, :, 0]
    matches = np.all(win == sig, axis = (2, 3, 4))
    ys, xs = np.where(matches)
    for y, x in zip(ys, xs):
        if y + h <= frame.shape[0] and x + w <= frame.shape[1]:
            if np.array_equal(frame[y:y + h, x:x + w], sprite):
                return int(x), int(y)
    raise AssertionError('sprite not found')


def count_template(frame: np.ndarray, sprite: np.ndarray) -> int:
    win = sliding_window_view(frame, sprite.shape)[:, :, 0]
    return int(np.all(win == sprite, axis = (2, 3, 4)).sum())


@pytest.mark.parametrize('cls', list(VIDEO_TASKS.values()))
def test_generate_shapes(cls):
    task = cls().generate(seed = 0)
    assert task['name'] == cls.name
    assert len(task['train']) == 3
    assert len(task['test']) == 1

    for level, first, clip in task['train'] + task['test']:
        assert isinstance(level, int)
        assert first.shape == (CANVAS, CANVAS, 3)
        assert clip.shape == (FRAMES, CANVAS, CANVAS, 3)
        assert first.dtype == clip.dtype == np.uint8


@pytest.mark.parametrize('cls', list(VIDEO_TASKS.values()))
def test_deterministic_by_seed(cls):
    a = cls().generate(seed = 42)
    b = cls().generate(seed = 42)
    c = cls().generate(seed = 43)

    assert a['name'] == b['name']
    for (la, fa, ca), (lb, fb, cb) in zip(a['train'] + a['test'], b['train'] + b['test']):
        assert la == lb
        assert np.array_equal(fa, fb)
        assert np.array_equal(ca, cb)

    differs = False
    for (_, fa, ca), (_, fc, cc) in zip(a['train'] + a['test'], c['train'] + c['test']):
        if not np.array_equal(fa, fc) or not np.array_equal(ca, cc):
            differs = True
            break
    assert differs


@pytest.mark.parametrize('cls', list(VIDEO_TASKS.values()))
def test_demo_and_test_level_ranges(cls):
    task = cls().generate(seed = 0, n_demos = 8, n_tests = 8)
    for level, _, _ in task['train']:
        lo, hi = cls.demo_levels
        assert lo <= level <= hi
    for level, _, _ in task['test']:
        lo, hi = cls.test_levels
        assert lo <= level <= hi


def test_translate_pan_train_levels_are_demo_levels_only():
    for seed in range(5):
        task = TranslatePan().generate(seed = seed, n_demos = 6, n_tests = 4)
        for level, _, _ in task['train']:
            lo, hi = TranslatePan.demo_levels
            assert lo <= level <= hi
            assert level == 1
        for level, _, _ in task['test']:
            lo, hi = TranslatePan.test_levels
            assert lo <= level <= hi
            assert level == 3


def test_identity_is_still():
    task = Identity().generate(seed = 0)
    for _, first, clip in task['train'] + task['test']:
        assert np.array_equal(clip[0], first)
        assert np.all(clip == first)


def test_translate_does_not_shift_background():
    task = Translate().generate(seed = 1)
    sprite = task['params']['sprite']

    for level, first, clip in task['train'] + task['test']:
        assert np.array_equal(clip[0], first)
        h, w = sprite.shape[:2]
        positions = [find_sprite(clip[t], sprite) for t in range(FRAMES)]
        for t in range(1, FRAMES):
            mask = np.ones((CANVAS, CANVAS), bool)
            x0, y0 = positions[0]
            xt, yt = positions[t]
            mask[y0:y0 + h, x0:x0 + w] = False
            mask[yt:yt + h, xt:xt + w] = False
            assert np.array_equal(clip[0][mask], clip[t][mask])

        origin = find_sprite(first, sprite)
        poses = trajectory(origin, (level * task['params']['dx'], level * task['params']['dy']), (0, 0))
        for t, (_, _, cam) in enumerate(poses):
            assert cam == (0, 0)
            assert find_sprite(clip[t], sprite) == poses[t][0]


def test_pan_does_not_change_sprite_world_pose():
    task = Pan().generate(seed = 2)
    sprite = task['params']['sprite']
    px, py = task['params']['px'], task['params']['py']

    for level, first, clip in task['train'] + task['test']:
        origin = find_sprite(first, sprite)
        poses = trajectory(origin, (0, 0), (level * px, level * py))
        worlds = []
        for t, ((sx, sy), world, _) in enumerate(poses):
            found = find_sprite(clip[t], sprite)
            assert found == (sx, sy)
            assert world == origin
            worlds.append(world)
        assert len(set(worlds)) == 1


def test_translate_pan_is_composition():
    rng = np.random.default_rng(3)
    params = TranslatePan().sample(rng)
    origin = (50, 50)
    params = dict(params, origin = origin)
    sprite = params['sprite']
    level = 1

    first_t, clip_t = Translate().render(rng, level, params)
    first_p, clip_p = Pan().render(rng, level, params)
    first_tp, clip_tp = TranslatePan().render(rng, level, params)

    assert find_sprite(first_t, sprite) == origin
    assert find_sprite(first_p, sprite) == origin
    assert find_sprite(first_tp, sprite) == origin

    for t in range(FRAMES):
        pt = find_sprite(clip_t[t], sprite)
        pp = find_sprite(clip_p[t], sprite)
        ptp = find_sprite(clip_tp[t], sprite)
        assert ptp == (pt[0] + pp[0] - origin[0], pt[1] + pp[1] - origin[1])

    # translate leaves a corner pixel unchanged; pan and translate_pan share bg phase
    assert np.all(clip_t[:, 0, 0] == clip_t[0, 0, 0])
    assert np.array_equal(clip_tp[:, 0, 0], clip_p[:, 0, 0])

    tv = (level * params['dx'], level * params['dy'])
    pv = (level * params['px'], level * params['py'])
    assert np.array_equal(clip_tp, blit_clip(sprite, origin, tv, pv, params['bg']))


def test_translate_pan_holdout_negates_pan():
    task = TranslatePan().generate(seed = 4, n_demos = 2, n_tests = 2)
    params = task['params']
    sprite = params['sprite']
    dx, dy, px, py = params['dx'], params['dy'], params['px'], params['py']

    for level, first, clip in task['train']:
        origin = find_sprite(first, sprite)
        poses = trajectory(origin, (level * dx, level * dy), (level * px, level * py))
        for t, ((sx, sy), _, _) in enumerate(poses):
            assert find_sprite(clip[t], sprite) == (sx, sy)

    for level, first, clip in task['test']:
        assert level == TranslatePan.test_levels[0]
        origin = find_sprite(first, sprite)
        poses = trajectory(origin, (level * dx, level * dy), (level * (-px), level * (-py)))
        for t, ((sx, sy), _, _) in enumerate(poses):
            assert find_sprite(clip[t], sprite) == (sx, sy)


def test_task_at_level_easy_demos_hard_query():
    task = task_at_level(TranslatePan, seed = 0, level = 3)
    demo_levels = [level for level, _, _ in task['train']]
    query_levels = [level for level, _, _ in task['test']]
    assert all(level == 1 for level in demo_levels)
    assert query_levels == [3]


def test_clip_stop_at_edge_no_wrap():
    rng = np.random.default_rng(0)
    params = Translate().sample(rng)
    sprite = params['sprite']
    origin = (CANVAS - SPRITE_SIZE, 40)
    first, clip = Translate().render(
        rng, 2, dict(params, origin = origin, dx = 1, dy = 0)
    )
    for t in range(FRAMES):
        x, y = find_sprite(clip[t], sprite)
        assert 0 <= x <= CANVAS - SPRITE_SIZE
        assert 0 <= y <= CANVAS - SPRITE_SIZE
        assert x == origin[0]
        assert y == origin[1]
    assert np.array_equal(clip[0], first)


def test_seed_extend_grows_and_distractor_stays():
    task = SeedExtend().generate(seed = 5)
    color = task['params']['color']
    distractor = task['params']['distractor']
    r0 = task['params']['r0']
    dr, dc = task['params']['distractor_pos']

    for level, first, clip in task['train'] + task['test']:
        assert np.array_equal(clip[0], first)
        assert np.all(clip[:, dr:dr + ANCHOR, dc:dc + ANCHOR] == distractor)
        for t in range(FRAMES):
            width = min(CANVAS, SEED_W + t * level)
            bar = clip[t, r0:r0 + BAR_H, 0:width]
            assert np.all(bar == color)
            if width < CANVAS:
                beyond = clip[t, r0:r0 + BAR_H, width:]
                assert not np.all(beyond == color)


def test_stamp_copy_stamps_anchors_and_is_static():
    task = StampCopy().generate(seed = 6)
    sprite = task['params']['sprite']

    for level, first, clip in task['train'] + task['test']:
        assert np.all(clip == clip[0])
        gray_mask = np.all(first == ANCHOR_GRAY, axis = -1)
        assert int(gray_mask.sum()) == level * ANCHOR * ANCHOR
        assert count_template(first, sprite) == 1
        assert count_template(clip[0], sprite) == 1 + level
        assert not np.all(clip[0][gray_mask] == ANCHOR_GRAY)


def test_recolor_maps_fill_and_is_static():
    task = Recolor().generate(seed = 7)
    fill_a, fill_b = task['params']['fill_a'], task['params']['fill_b']
    sprite = task['params']['sprite']

    for _, first, clip in task['train'] + task['test']:
        assert np.all(clip == clip[0])
        assert not np.array_equal(first, clip[0])
        origin = find_sprite(first, sprite)
        out_patch = clip[0][origin[1]:origin[1] + SPRITE_SIZE, origin[0]:origin[0] + SPRITE_SIZE]
        src_fill = np.all(sprite == fill_a, axis = -1)
        assert np.all(out_patch[src_fill] == fill_b)
        assert not np.any(np.all(out_patch == fill_a, axis = -1))
