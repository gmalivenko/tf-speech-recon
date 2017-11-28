#!/usr/bin/env python

import os

print("CUDA_VISIBLE_DEVICES = %r" % os.environ.get("CUDA_VISIBLE_DEVICES"))

import pycuda
try:
    import pycuda.driver
except ImportError:
    print("LD_LIBRARY_PATH = %r" % os.environ["LD_LIBRARY_PATH"])
    raise

pycuda.driver.init()
count = pycuda.driver.Device.count()
print "Count:", count

devs = [pycuda.driver.Device(i) for i in range(count)]
for i, dev in enumerate(devs):
	try:
		ctx = dev.make_context()
	except pycuda.driver.LogicError:
		print "Dev %i: Cannot make context" % i
	else:
		print "Dev %i: Is free" % i
		ctx.detach()

