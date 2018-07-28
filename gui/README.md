## Development setup

Qt5 is used to make the interface.

```bash
sudo apt install qt5-default
```

Other needed tools are `cmake` and `build-essential`.

## EGL and GL library errors

CUDA installation might mess up Mesa GL and EGL dependencies, where

```bash
/usr/lib/x86_64-linux-gnu/libGL.so
/usr/lib/x86_64-linux-gnu/libEGL.so
```

Would link to Mesa libraries under `mesa/` which would link to
specific libraries that don't exist. The solution is

```bash
sudo rm /usr/lib/x86_64-linux-gnu/libGL.so
sudo rm /usr/lib/x86_64-linux-gnu/libGL.so
sudo ln /usr/lib/x86_64-linux-gnu/libEGL.so.1 /usr/lib/x86_64-linux-gnu/libEGL.so
sudo ln /usr/lib/x86_64-linux-gnu/libEGL.so.1 /usr/lib/x86_64-linux-gnu/libEGL.so
```

So that CMake can find the libraries.
