import cv2
import numpy
import os
import binascii
import array
import scipy.misc
import subprocess
import time
start_time = time.time()

count =0;
directory = r'path of the Malware binaries'
for filename in sorted(os.listdir(directory)):
	subprocess.Popen('insert path of linux script prepare_spectrograme.sh %s %s' % ('path of the Malware binaries'+filename,'path of the spectrogram images that will be generated'+ str(count)),shell=True)
	count=count+1
