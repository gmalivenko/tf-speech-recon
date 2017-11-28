#!/usr/bin/env python

import sys
import pycuda
import pycuda.driver

pycuda.driver.init()
count = pycuda.driver.Device.count()

devs = [pycuda.driver.Device(i) for i in range(count)]
for i, dev in enumerate(devs):
    try:
        ctx = dev.make_context()
    except pycuda.driver.LogicError:
        continue
    else:
        print(i)
        ctx.detach()
        sys.exit(0)

sys.exit(1)
