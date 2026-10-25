# C host example

This example uses only `include/swxbx.h` and the release shared library.
It creates an instance with memory callbacks backed by a fake buffer. The
present callback writes the RGBA frame as a PPM image. The program resolves
functions by name, records a clear, submits it, presents the image, writes the
census and shapes reports, checks the last error after an invalid function
call, and destroys the instance.

Run `make run` from this directory. The command writes `census.txt`,
`shapes.txt`, and `frame.ppm` in the current directory.
