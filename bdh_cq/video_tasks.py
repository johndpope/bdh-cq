"""Synthetic video oracles for BDH-CQ video generation (v0).

Seven families: identity, translate, pan, translate_pan, seed_extend,
stamp_copy, recolor. Demos draw `demo_levels`, held-out queries
`test_levels`. Deterministic given a seed.

Canvas is 22 x 128 x 128 RGB. Sprite motion is integer px/frame, no wrap;
if the next step would put any sprite pixel off-canvas, hold the last
legal pose.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.random import Generator

# locked geometry

CANVAS = 128
FRAMES = 22
SPRITE_SIZE = 32  # 2x2 FakeVAE cells (spatial 16). 24px straddled the grid.
STAMP_SIZE = 16
BAR_H = 16
ANCHOR = 8
CHECKER = 32
SEED_W = 8
SPRITE_ALIGN = 16  # VAE_SPATIAL; origins snap so a sprite fills whole cells
ANCHOR_GRAY = np.array([160, 160, 160], np.uint8)
BG_LO, BG_HI = 8, 48  # dim checker: pan still reads, sprite wins avg-pool
SPRITE_LO, SPRITE_HI = 180, 255

UNITS = tuple(
    (dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy
)


# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def randint(rng: Generator, lo: int, hi: int) -> int:
    return int(rng.integers(lo, hi + 1))

def sample_unit(rng: Generator) -> tuple[int, int]:
    return UNITS[int(rng.integers(0, len(UNITS)))]

def sample_color(
    rng: Generator,
    forbidden: tuple = (),
    lo: int = 0,
    hi: int = 256,
) -> np.ndarray:
    for _ in range(64):
        color = rng.integers(lo, hi, 3, dtype = np.uint8)
        if all(not np.array_equal(color, f) for f in forbidden):
            return color
    return rng.integers(lo, hi, 3, dtype = np.uint8)

def sample_bg(rng: Generator) -> dict[str, np.ndarray]:
    forbidden = (ANCHOR_GRAY,)
    top = sample_color(rng, forbidden, BG_LO, BG_HI)
    bot = sample_color(rng, forbidden + (top,), BG_LO, BG_HI)
    chk_a = sample_color(rng, forbidden + (top, bot), BG_LO, BG_HI)
    chk_b = sample_color(rng, forbidden + (top, bot, chk_a), BG_LO, BG_HI)
    return dict(top = top, bot = bot, chk_a = chk_a, chk_b = chk_b)

def make_sprite(
    rng: Generator,
    size: int = SPRITE_SIZE,
    fill: np.ndarray | None = None,
    forbidden: tuple = ()
) -> tuple[np.ndarray, np.ndarray]:
    forbidden = forbidden + (ANCHOR_GRAY,)
    fill = default(fill, sample_color(rng, forbidden, SPRITE_LO, SPRITE_HI))
    sprite = np.broadcast_to(fill, (size, size, 3)).copy()
    n_blobs = randint(rng, 2, 3)
    for _ in range(n_blobs):
        color = sample_color(rng, forbidden + (fill,), SPRITE_LO, SPRITE_HI)
        bh, bw = randint(rng, 3, max(3, size // 4)), randint(rng, 3, max(3, size // 4))
        y = randint(rng, 2, size - bh - 2)
        x = randint(rng, 2, size - bw - 2)
        sprite[y:y + bh, x:x + bw] = color
    return sprite, fill

def draw_background(phase_x: int, phase_y: int, bg: dict[str, np.ndarray], canvas: int = CANVAS) -> np.ndarray:
    y, x = np.indices((canvas, canvas))
    wx = x + int(phase_x)
    wy = y + int(phase_y)
    period = 256
    t = (wy % period).astype(np.float64) / period
    lerp = (1.0 - t)[..., None] * bg['top'].astype(np.float64) + t[..., None] * bg['bot'].astype(np.float64)
    cx = (wx // CHECKER) & 1
    cy = (wy // CHECKER) & 1
    chk = np.where(
        (cx ^ cy)[..., None],
        bg['chk_a'].astype(np.float64),
        bg['chk_b'].astype(np.float64)
    )
    return np.clip(0.7 * lerp + 0.3 * chk, 0, 255).astype(np.uint8)

def in_bounds(x: int, y: int, size: int = SPRITE_SIZE, canvas: int = CANVAS) -> bool:
    return x >= 0 and y >= 0 and x + size <= canvas and y + size <= canvas

def sample_origin_1d(rng: Generator, vel: int, n_steps: int, size: int, canvas: int) -> int:
    lo, hi = 0, canvas - size
    travel = int(vel) * n_steps
    if travel > 0:
        hi = min(hi, canvas - size - travel)
    elif travel < 0:
        lo = max(lo, -travel)
    if lo > hi:
        return 0 if travel > 0 else canvas - size
    lo_a = ((lo + SPRITE_ALIGN - 1) // SPRITE_ALIGN) * SPRITE_ALIGN
    hi_a = (hi // SPRITE_ALIGN) * SPRITE_ALIGN
    if lo_a > hi_a:
        snapped = (lo // SPRITE_ALIGN) * SPRITE_ALIGN
        return int(np.clip(snapped, 0, canvas - size))
    return randint(rng, 0, (hi_a - lo_a) // SPRITE_ALIGN) * SPRITE_ALIGN + lo_a

def sample_origin(
    rng: Generator,
    screen_vel: tuple[int, int],
    size: int = SPRITE_SIZE,
    canvas: int = CANVAS,
    n_steps: int = FRAMES - 1
) -> tuple[int, int]:
    vx, vy = screen_vel
    return (
        sample_origin_1d(rng, vx, n_steps, size, canvas),
        sample_origin_1d(rng, vy, n_steps, size, canvas)
    )

def trajectory(
    origin: tuple[int, int],
    translate: tuple[int, int],
    pan: tuple[int, int],
    n_frames: int = FRAMES,
    size: int = SPRITE_SIZE,
    canvas: int = CANVAS
) -> list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]:
    """per-frame (screen, world, camera). clip/stop at the canvas edge."""

    wx, wy = int(origin[0]), int(origin[1])
    cx, cy = 0, 0
    vx, vy = int(translate[0]), int(translate[1])
    px, py = int(pan[0]), int(pan[1])
    out = []
    for _ in range(n_frames):
        out.append(((wx - cx, wy - cy), (wx, wy), (cx, cy)))
        nwx, nwy = wx + vx, wy + vy
        ncx, ncy = cx + px, cy + py
        if in_bounds(nwx - ncx, nwy - ncy, size, canvas):
            wx, wy, cx, cy = nwx, nwy, ncx, ncy
    return out

def blit(dst: np.ndarray, src: np.ndarray, x: int, y: int) -> None:
    h, w = src.shape[:2]
    dst[y:y + h, x:x + w] = src

def blit_clip(
    sprite: np.ndarray,
    origin: tuple[int, int],
    translate: tuple[int, int],
    pan: tuple[int, int],
    bg: dict[str, np.ndarray],
    n_frames: int = FRAMES,
    canvas: int = CANVAS
) -> np.ndarray:
    """(T, H, W, 3) uint8. camera pan shifts bg phase; sprite is world-posed."""

    size = sprite.shape[0]
    poses = trajectory(origin, translate, pan, n_frames, size, canvas)
    frames = np.empty((n_frames, canvas, canvas, 3), np.uint8)
    for t, ((sx, sy), _, (cx, cy)) in enumerate(poses):
        frames[t] = draw_background(cx, cy, bg, canvas)
        blit(frames[t], sprite, sx, sy)
    return frames


# base task

class VideoTask(ABC):
    """task dicts hold {"name", "params", "train", "test"}, where examples
    are (level, first_frame, clip) with clip shape (22, 128, 128, 3)
    """

    name: str
    demo_levels: tuple[int, int]
    test_levels: tuple[int, int]

    @abstractmethod
    def sample(self, rng: Generator) -> dict[str, Any]:
        ...

    @abstractmethod
    def render(self, rng: Generator, level: int, params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        ...

    def query_params(self, params: dict[str, Any]) -> dict[str, Any]:
        return params

    def generate(
        self,
        seed: int = 0,
        n_demos: int = 3,
        n_tests: int = 1,
        test_levels: tuple[int, int] | None = None
    ) -> dict[str, Any]:
        rng = np.random.default_rng(seed)
        params = self.sample(rng)

        train = []
        for _ in range(n_demos):
            level = randint(rng, *self.demo_levels)
            train.append((level, *self.render(rng, level, params)))

        test = []
        qparams = self.query_params(params)
        for _ in range(n_tests):
            level = randint(rng, *(test_levels or self.test_levels))
            test.append((level, *self.render(rng, level, qparams)))

        return dict(name = self.name, params = params, train = train, test = test)


def _moving_pair(
    rng: Generator,
    params: dict[str, Any],
    translate: tuple[int, int],
    pan: tuple[int, int],
    sprite: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    sprite = default(sprite, params['sprite'])
    size = sprite.shape[0]
    origin = params.get('origin')
    if origin is None:
        origin = sample_origin(rng, (translate[0] - pan[0], translate[1] - pan[1]), size)
    clip = blit_clip(sprite, origin, translate, pan, params['bg'])
    return clip[0].copy(), clip


# families

class Identity(VideoTask):
    """output = repeat(first_frame, 22). Sprite at a random legal (x, y)."""

    name = 'identity'
    demo_levels = (0, 0)
    test_levels = (0, 0)

    def sample(self, rng):
        sprite, fill = make_sprite(rng)
        return dict(sprite = sprite, fill = fill, bg = sample_bg(rng))

    def render(self, rng, level, params):
        return _moving_pair(rng, params, (0, 0), (0, 0))


class Translate(VideoTask):
    """sprite += level * (dx, dy) px/frame. camera / bg static."""

    name = 'translate'
    demo_levels = (1, 2)
    test_levels = (3, 4)

    def sample(self, rng):
        sprite, fill = make_sprite(rng)
        dx, dy = sample_unit(rng)
        return dict(sprite = sprite, fill = fill, bg = sample_bg(rng), dx = dx, dy = dy)

    def render(self, rng, level, params):
        vx, vy = level * params['dx'], level * params['dy']
        return _moving_pair(rng, params, (vx, vy), (0, 0))


class Pan(VideoTask):
    """bg phase shifts level * (px, py) px/frame; sprite world-static."""

    name = 'pan'
    demo_levels = (1, 2)
    test_levels = (3, 4)

    def sample(self, rng):
        sprite, fill = make_sprite(rng)
        px, py = sample_unit(rng)
        return dict(sprite = sprite, fill = fill, bg = sample_bg(rng), px = px, py = py)

    def render(self, rng, level, params):
        px, py = level * params['px'], level * params['py']
        return _moving_pair(rng, params, (0, 0), (px, py))


class TranslatePan(VideoTask):
    """translate and pan, independent units. query pan sign is negated."""

    name = 'translate_pan'
    demo_levels = (1, 1)
    test_levels = (3, 3)

    def sample(self, rng):
        sprite, fill = make_sprite(rng)
        dx, dy = sample_unit(rng)
        px, py = sample_unit(rng)
        return dict(sprite = sprite, fill = fill, bg = sample_bg(rng), dx = dx, dy = dy, px = px, py = py)

    def query_params(self, params):
        out = dict(params)
        out['px'] = -int(params['px'])
        out['py'] = -int(params['py'])
        return out

    def render(self, rng, level, params):
        vx, vy = level * params['dx'], level * params['dy']
        px, py = level * params['px'], level * params['py']
        return _moving_pair(rng, params, (vx, vy), (px, py))


class SeedExtend(VideoTask):
    """16 px-tall bar on the left; grows level px/frame; distractor stays."""

    name = 'seed_extend'
    demo_levels = (1, 2)
    test_levels = (3, 4)

    def sample(self, rng):
        r0 = randint(rng, 0, CANVAS - BAR_H)
        color = sample_color(rng, (ANCHOR_GRAY,))
        distractor = sample_color(rng, (ANCHOR_GRAY, color))
        free = [r for r in range(CANVAS - ANCHOR + 1) if r + ANCHOR <= r0 or r >= r0 + BAR_H]
        dr = int(rng.choice(free))
        dc = randint(rng, CANVAS // 2, CANVAS - ANCHOR)
        return dict(
            bg = sample_bg(rng),
            r0 = r0,
            color = color,
            distractor = distractor,
            distractor_pos = (dr, dc)
        )

    def render(self, rng, level, params):
        r0, color, distractor = params['r0'], params['color'], params['distractor']
        dr, dc = params['distractor_pos']
        bg = draw_background(0, 0, params['bg'])
        frames = np.empty((FRAMES, CANVAS, CANVAS, 3), np.uint8)
        for t in range(FRAMES):
            frames[t] = bg
            width = min(CANVAS, SEED_W + t * level)
            frames[t, r0:r0 + BAR_H, 0:width] = color
            frames[t, dr:dr + ANCHOR, dc:dc + ANCHOR] = distractor
        return frames[0].copy(), frames


class StampCopy(VideoTask):
    """level = #anchors. 16x16 sprite; 8x8 gray anchors; output stamps, static."""

    name = 'stamp_copy'
    demo_levels = (1, 2)
    test_levels = (3, 4)

    def sample(self, rng):
        sprite, fill = make_sprite(rng, STAMP_SIZE)
        return dict(sprite = sprite, fill = fill, bg = sample_bg(rng))

    def render(self, rng, level, params):
        sprite = params['sprite']
        cells = [
            (c, r)
            for r in range(0, CANVAS - STAMP_SIZE + 1, STAMP_SIZE)
            for c in range(0, CANVAS - STAMP_SIZE + 1, STAMP_SIZE)
        ]
        rng.shuffle(cells)
        src = cells[0]
        anchors = cells[1:1 + level]

        bg = draw_background(0, 0, params['bg'])
        hit = np.all(bg == ANCHOR_GRAY, axis = -1)
        if hit.any():
            bg = bg.copy()
            bg[hit] = (int(ANCHOR_GRAY[0]) ^ 1, int(ANCHOR_GRAY[1]), int(ANCHOR_GRAY[2]))
        first = bg.copy()
        blit(first, sprite, *src)
        for ax, ay in anchors:
            first[ay:ay + ANCHOR, ax:ax + ANCHOR] = ANCHOR_GRAY

        out = bg.copy()
        blit(out, sprite, *src)
        for ax, ay in anchors:
            blit(out, sprite, ax, ay)

        clip = np.repeat(out[None, ...], FRAMES, axis = 0)
        return first, clip


class Recolor(VideoTask):
    """sprite fill A -> B; no motion. mapping lives in params."""

    name = 'recolor'
    demo_levels = (1, 1)
    test_levels = (1, 1)

    def sample(self, rng):
        fill_a = sample_color(rng, (ANCHOR_GRAY,))
        fill_b = sample_color(rng, (ANCHOR_GRAY, fill_a))
        sprite, _ = make_sprite(rng, fill = fill_a, forbidden = (fill_b,))
        return dict(sprite = sprite, fill_a = fill_a, fill_b = fill_b, bg = sample_bg(rng))

    def render(self, rng, level, params):
        out_sprite = params['sprite'].copy()
        mask = np.all(out_sprite == params['fill_a'], axis = -1)
        out_sprite[mask] = params['fill_b']
        origin = params.get('origin')
        if origin is None:
            origin = sample_origin(rng, (0, 0))
        first_clip = blit_clip(params['sprite'], origin, (0, 0), (0, 0), params['bg'])
        out_clip = blit_clip(out_sprite, origin, (0, 0), (0, 0), params['bg'])
        return first_clip[0].copy(), out_clip


VIDEO_TASKS = dict(
    identity = Identity,
    translate = Translate,
    pan = Pan,
    translate_pan = TranslatePan,
    seed_extend = SeedExtend,
    stamp_copy = StampCopy,
    recolor = Recolor
)


def task_at_level(
    cls,
    seed: int,
    level: int,
    n_demos: int = 3,
    n_tests: int = 1
) -> dict[str, Any]:
    # demos from the easy range, query at exactly `level`
    return cls().generate(seed, n_demos, n_tests, test_levels = (level, level))
