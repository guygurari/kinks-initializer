#!/usr/bin/env python3

import sys
import tfutils
import logging

# tf.logging.set_verbosity(tf.logging.WARN)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

filename = sys.argv[1]
tfutils.read_events_file(filename)
