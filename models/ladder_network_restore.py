import tensorflow as tf

sess=tf.Session()
saver = tf.train.import_meta_graph('/var/folders/4j/81fnzhx15t9fp6h3t98dx5zr0000gn/T/tmpLTSOHf/model.ckpt-1800')
saver.restore(sess,tf.train.latest_checkpoint('./'))

graph = tf.get_default_graph()