"""
Make a video from the combined images with ffmpeg.

    python main.py video                      # images of config.COMBINED -> mp4
    python main.py video --fps 5 --output temp.mp4
    python main.py video --indir plots/temp --name cbox.temp   # any numbered images
    python main.py combined --video           # plot, then make the video

Takes ``<indir>/<name>.<axis>.<NNNNN>.<format>`` in frame order and writes
``<indir>/<name>.<axis>.mp4``.  Defaults come from config.COMBINED and
config.PLOT.  The encoder is libx264 if ffmpeg has it, else NVIDIA's
h264_nvenc, else mpeg4.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

if "." in __package__:  # imported as vis_athenak.plotting
    from .. import config
else:  # run from the repo root
    import config

# Encoders in order of preference, with their quality settings.
CODECS = {
    "libx264": ["-c:v", "libx264", "-crf", "18", "-preset", "medium"],
    "h264_nvenc": ["-c:v", "h264_nvenc", "-cq", "19", "-preset", "p5"],
    "mpeg4": ["-c:v", "mpeg4", "-q:v", "2"],
}


def find_images(indir: Path, stem: str, image_format: str = "png") -> list[Path]:
    """``<stem>.<NNNNN>.<format>`` files in ``indir``, sorted by frame number."""
    pattern = re.compile(rf"^{re.escape(stem)}\.(\d+)\.{re.escape(image_format)}$")
    if not Path(indir).is_dir():
        return []
    found = [(int(m.group(1)), p) for p in Path(indir).iterdir()
             if (m := pattern.match(p.name))]
    return [p for _, p in sorted(found)]


def make_video(images: list[Path], output, fps: float = 10, codec: str = "auto") -> Path:
    """Encode ``images`` (same format and size), in order, into ``output``."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found; install it (e.g. sudo apt install ffmpeg)")
    if not images:
        raise FileNotFoundError("no images to make a video of")
    images, output = [Path(p) for p in images], Path(output)
    image_format = images[0].suffix.lstrip(".")

    if codec == "auto":
        encoders = subprocess.run([ffmpeg, "-hide_banner", "-encoders"],
                                  capture_output=True, text=True).stdout
        candidates = [c for c in CODECS if re.search(rf"\s{c}\s", encoders)]
    else:
        candidates = [codec]

    # Link the images as 00000, 00001, ... so gaps in frame numbers don't matter.
    with tempfile.TemporaryDirectory() as tmp:
        for i, image in enumerate(images):
            (Path(tmp) / f"{i:05d}.{image_format}").symlink_to(image.resolve())
        errors = []
        for name in candidates:
            cmd = [ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps),
                   "-i", str(Path(tmp) / f"%05d.{image_format}"),
                   # Even width/height and yuv420p, so every player can open it.
                   "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p",
                   *CODECS.get(name, ["-c:v", name]), str(output)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return output
            errors.append(f"{name}: {result.stderr.strip().splitlines()[-1:]}")
    raise RuntimeError("ffmpeg failed with every encoder:\n  " + "\n  ".join(errors))


def main(argv=None):
    plot, combined = config.PLOT, config.COMBINED
    parser = argparse.ArgumentParser(description="Make a video of numbered images with ffmpeg.")
    parser.add_argument("--indir", help="folder of the images (default: the combined "
                                        "images' folder for config.COMBINED)")
    parser.add_argument("--name", default=combined["name"],
                        help=f"image name prefix (default: {combined['name']})")
    parser.add_argument("--axis", default=plot["axis"],
                        help=f"slice axis in the image names (default: {plot['axis']})")
    parser.add_argument("--format", default=plot["format"], help="image format")
    parser.add_argument("--fps", type=float, default=combined["fps"],
                        help=f"frames per second (default: {combined['fps']})")
    parser.add_argument("--output", help="video file (default: <indir>/<name>.<axis>.mp4)")
    parser.add_argument("--run", default=config.RUN, choices=list(config.SIMULATIONS),
                        help="run of layout cells that name no run, for the default "
                             f"--indir (default: {config.RUN})")
    parser.add_argument("--codec", default="auto",
                        help=f"ffmpeg encoder (default: first available of {list(CODECS)})")
    args = parser.parse_args(argv)

    if args.indir:
        indir = Path(args.indir)
    else:
        from .combined import default_outdir, normalize_layout
        try:
            indir = default_outdir(normalize_layout(combined["layout"], args.run))
        except RuntimeError as err:
            parser.error(f"{err}, or pass --indir")
    try:
        stem = f"{args.name}.{args.axis}"
        images = find_images(indir, stem, args.format)
        if not images:
            raise FileNotFoundError(f"no images {stem}.NNNNN.{args.format} in {indir}")
        out = make_video(images, args.output or indir / f"{stem}.mp4", fps=args.fps,
                         codec=args.codec)
    except (FileNotFoundError, RuntimeError) as err:
        parser.error(str(err))
    print(f"video: {out}")


if __name__ == "__main__":
    main()
