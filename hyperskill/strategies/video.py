"""
Omni multiple-choice options (evaluation only, all off by default).

  time_grounding   the video length and the time of every keyframe after the
                   question, the shared audio timeline, and the thirds
                   convention for beginning / middle / end.
  option_check     check every option against the frames and the audio,
                   eliminate contradicted ones, a hedge option ("not sure",
                   "unknown", ...) only if all others are contradicted; one
                   option letter in <answer>. Same single call.
"""

from __future__ import annotations

from typing import List, Tuple

from .voting import parse_options


def video_info(video_path: str) -> Tuple[int, float]:
    import cv2
    cap = cv2.VideoCapture(video_path)
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        cap.release()


def frame_indices(total: int, n: int) -> List[int]:
    """Same positions as trainer._extract_keyframes (int(i * total / n))."""
    return [min(total - 1, int(i * total / n)) for i in range(n)]


def mmss(t: float) -> str:
    t = max(0, int(round(t)))
    return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60}:{t % 60:02d}"


def time_text(total: int, fps: float, n: int) -> str:
    if total <= 0 or fps <= 0:
        return ""
    dur = total / fps
    times = ", ".join(f"image {i + 1} = {mmss(k / fps)}" for i, k in enumerate(frame_indices(total, n)))
    return (f"Video timing: the video lasts {mmss(dur)}. The {n} images are keyframes in temporal order, taken "
            f"at {times}. The audio track covers the whole video on the same timeline (0:00 to {mmss(dur)}). "
            f"For questions about when something happens, compare its time with the total length: the beginning "
            f"is the first third, the middle the second third and the end the last third. Between two keyframes, "
            f"rely on the audio and the neighbouring frames.")


OE_LINE = ("Before answering, check every option against what the frames show and what the audio says: for each "
           "option write one short line 'X: supported / contradicted / no evidence' with the frame time or the "
           "audio it relies on, then choose the best-supported option. Choose an option such as 'not sure', "
           "'unknown', 'it is unclear' or 'none of the above' only if every other option is contradicted. Never "
           "decline to answer: always end with exactly one option letter inside <answer>...</answer>, for example "
           "<answer>B</answer>.")


def video_timing(trainer, item) -> str:
    """time_grounding text of a video item ("" when off or not a video item)."""
    if not (trainer.cfg.hypergraph.time_grounding and item.video_path and not item.images):
        return ""
    total, fps = video_info(item.video_path)
    return time_text(total, fps, trainer.cfg.rollout.max_frames)


def prompt_suffix(trainer, item) -> str:
    """time_grounding / option_check lines after the question ("" when both off)."""
    parts = []
    t = video_timing(trainer, item)
    if t:
        parts.append(t)
    if trainer.cfg.hypergraph.option_check and parse_options(item.question):
        parts.append(OE_LINE)
    return " ".join(parts)
