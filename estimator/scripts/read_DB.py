 #!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import scipy.misc

sess = tf.Session()
stop = 20000

for event in tf.train.summary_iterator('trained_model/events.out.tfevents.1531269146.tensor2'):
	if event.step != 0 and event.step % 2 != 0:

		if event.step > stop:
			exit()

		for value in event.summary.value:

			if value.tag == 'predictions/predictions/fake_y/image/0':
				image = tf.image.decode_png(value.image.encoded_image_string, channels = 1, dtype = tf.uint8)
				numpy_image = sess.run(image)
				numpy_image = np.squeeze(numpy_image)
				print('Saving image_%i.jpg' % event.step)
				scipy.misc.imsave('patches/image_%i.jpg' % event.step, numpy_image)

