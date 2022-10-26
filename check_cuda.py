#!/usr/bin/env python

"""Create a pip.conf using the correct cuda version
"""
import functools
import logging
import os
import re
import shutil
import sys

logging.getLogger().setLevel(logging.INFO)

@functools.cache
def get_cuda_version() -> float:
    """Get the CUDA version of the first GPU

    Returns:
        float: The cuda version of the first GPU. Or nan, if no GPUs present
    """
    try:
        result = os.popen(f"nvidia-smi -q 2> {os.devnull}").read()
        version = float(
            re.findall(r"CUDA Version\s*:\s*([0-9]+\.[0-9]+)", result)[0]
        )
        return version
    except IndexError:
        return float("nan")


def get_pip_conf():
    return os.path.join(
        sys.prefix,
        "pip."
        + (
            "ini"
            if sys.platform.startswith("win")
            or (sys.platform == "cli" and os.name == "nt")
            else "conf"
        ),
    )


def main():
    if get_cuda_version() == float("nan"):
        exit()

    try:
        import jax

        print("Jax already installed")
    except ModuleNotFoundError:
        try:
            conf_location = get_pip_conf()
            append_at = -1
            lines: list[str] | None = None
            if os.path.exists(conf_location):
                with open(conf_location, "r") as fp:
                    lines = fp.readlines()
                    for line_no, line in enumerate(lines):
                        if (
                            re.match("^find-links", line)
                            and append_at != -1
                        ):
                            append_at = line_no
                        if "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html" in line:
                            exit()
            if append_at >= 0 and lines is not None:
                logging.info(
                    f"""Updating pip.conf to include extra link for jax cuda.
                    Backing up the file to '{conf_location}.backup'
                    
                    
                    To restore, run:
                    
                    mv '{conf_location}.backup' '{conf_location}'

                    '"""
                )
                shutil.copyfile(conf_location, conf_location + ".backup")
                lines.insert(
                    append_at + 1,
                    f"\n\thttps://storage.googleapis.com/jax-releases/jax_cuda_releases.html",
                )
                with open(conf_location, "w") as fp:
                    fp.writelines(lines)
            else:
                logging.info(
                    f"""
                    Creating a pip configuration file to include cuda links for jax.

                    To restore, original state, run:

                    rm '{conf_location}'
                    """
                )
                with open(conf_location, "w") as fp:
                    fp.write(
                        f"""[global]
find-links = https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"""
                    )

        except ValueError as e:
            logging.debug(e)


if __name__ == "__main__":
    main()
