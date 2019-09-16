def extras():
        c = tf.placeholder(tf.int32, shape=[None, 3, 100])
        print(c.shape)
        [d, e, f] = tf.unstack(c, axis=1) #unstack tensors along one dimension.
        print(d.shape)
        print(e.shape)

        a = tf.concat([d,e],1) #Concatenates tensors along one dimension.
        print(a.shape)

        b = tf.stack([d,e],1)
        print(b.shape)