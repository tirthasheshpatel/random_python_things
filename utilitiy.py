#!/usr/bin/env python

import os
import sys
import time


class colors:
  NOCOLOR='\033[0m'
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  ORANGE='\033[0;33m'
  BLUE='\033[0;34m'
  PURPLE='\033[0;35m'
  CYAN='\033[0;36m'
  YELLOW='\033[1;33m'


if __name__ == "__main__":
  sys.stderr.write("Wtf, It's working!\n")
  l = ["\\", "|", "/", "-"]
  c = [colors.ORANGE,
       colors.GREEN,
       colors.BLUE,
       colors.PURPLE,
       colors.CYAN,
       colors.YELLOW]
  i=0
  while i<100:
    sys.stderr.write("\r%sLoading: %s%s" % (c[i%6], l[i%4], colors.NOCOLOR))
    i+=1
    time.sleep(0.1)
  sys.stderr.flush()
  sys.stderr.write("\r%sHey human.%s" % (colors.GREEN, colors.NOCOLOR))
  time.sleep(2)
  sys.stdout.flush()
  sys.stderr.write("\r" + " "*50)
  sys.stderr.write("\r%sIt's getting late!%s" % (colors.CYAN, colors.NOCOLOR))
  time.sleep(2)
  sys.stderr.flush()
  sys.stderr.write("\r" + " "*50)
  sys.stderr.write("\r%sGood night human!%s" % (colors.YELLOW, colors.NOCOLOR))
  time.sleep(2)
  sys.stderr.flush()
  sys.stderr.write("\r" + " "*50)
  sys.stderr.write("\r%sbye bye...%s" % (colors.ORANGE, colors.NOCOLOR))
  time.sleep(2)
  sys.stderr.flush()
  sys.stderr.write("\r" + " "*50)
  sys.stderr.write("\r%sError loading!%s\n" % (colors.RED, colors.NOCOLOR))
