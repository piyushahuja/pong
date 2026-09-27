"""
In-notebook replay of a trained policy.

    from replay import watch
    watch(seed=1)          # returns an HTML5 player, inline, no ffmpeg needed

Frames are captured with render_mode="rgb_array" and animated with
matplotlib's to_jshtml, which embeds the frames as base64 PNGs plus a
JS scrubber. That keeps the output self-contained: it survives a kernel
restart and nbconvert. It does NOT survive GitHub, which strips the script.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

from pong import checkpoints, env as pong_env, evaluate


def watch(
    seed=1,
    greedy=True,
    frame_stride=3,
    max_steps=20000,
    checkpoint=None,
    show_prob=True,
    fps=30,
):
    """Play one episode and return an inline HTML5 animation."""
    policy = checkpoints.load_policy(checkpoint)
    env = pong_env.make_env("rgb_array")
    try:
        us, them, frames, probs = evaluate.rollout(
            policy, env,
            greedy=greedy,
            seed=seed,
            max_steps=max_steps,
            collect_frames=True,
            frame_stride=frame_stride,
        )
    finally:
        env.close()

    return animate(frames, probs, title=f"{us} - {them}", show_prob=show_prob, fps=fps)


def animate(frames, probs, title="", show_prob=True, fps=30):
    """
    Frames on the left; if show_prob, a P(UP) meter on the right that moves
    in step with the game. Seeing the meter slam to 0 or 1 as the ball
    approaches is the clearest view of what the policy has actually learnt.
    """
    if show_prob:
        fig, (ax_game, ax_p) = plt.subplots(
            1, 2, figsize=(7, 4.4), gridspec_kw={"width_ratios": [3, 1]}
        )
    else:
        fig, ax_game = plt.subplots(figsize=(4.2, 5))
        ax_p = None

    fig.patch.set_facecolor("#f4f7f6")

    im = ax_game.imshow(frames[0])
    ax_game.axis("off")
    ax_game.set_title(title, color="#1a2b2e", fontsize=11)

    if ax_p is not None:
        bar = ax_p.barh([0], [probs[0]], color="#2e7d8f", height=0.5)[0]
        ax_p.axvline(0.5, color="#1a2b2e", lw=0.8, ls=":")
        ax_p.set_xlim(0, 1)
        ax_p.set_ylim(-1, 1)
        ax_p.set_yticks([])
        ax_p.set_xticks([0, 0.5, 1])
        ax_p.set_xlabel("P(UP)", color="#1a2b2e", fontsize=9)
        ax_p.set_facecolor("#f4f7f6")
        for s in ax_p.spines.values():
            s.set_visible(False)
        label = ax_p.text(0.5, 0.6, "", ha="center", fontsize=10, color="#1a2b2e")

    def update(i):
        im.set_data(frames[i])
        if ax_p is not None:
            bar.set_width(probs[i])
            bar.set_color("#2e7d8f" if probs[i] > 0.5 else "#c25b56")
            label.set_text("UP" if probs[i] > 0.5 else "DOWN")
            return im, bar, label
        return (im,)

    anim = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=1000 / fps, blit=False
    )
    plt.close(fig)
    return HTML(anim.to_jshtml(fps=fps))
