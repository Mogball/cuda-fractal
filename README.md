### CUDA Installation on Ubuntu

Download the CUDA runfile, e.g. `cuda_9.2_linux.run`.

```bash
runfile=cuda_9.2_linux.run
chmod +x ${runfile}
./${runfile}
```

Follow prompts and install driver if needed. Then update
`PATH` and `LD_LIBRARY_PATH` to include the locations specified
by the installer.

### Driver Installation

Disable the `nouveau` driver and install the `nvidia` driver.
Create file `/etc/modprobe.d/blacklist-nouveau.conf` with the
contents

```
blacklist nouveau
options nouveau modeset=0
```

And then run `sudo update-initramfs -u`. Install the `nvidia-*`
driver and reboot. Ensure that the driver is running with
`lshw -c video` and `modinfo nvidia`.
