from buffer import *
from data import *
from layers import *
from sample import *


def refiner(scope, input, reuse=False):
    with tf.variable_scope(scope, reuse=reuse) as scp:
        output = conv(input, 64, 3, 1, "conv1")
        output = res_block(output, 64, 3, 1, "block1")
        output = res_block(output, 64, 3, 1, "block2")
        output = res_block(output, 64, 3, 1, "block3")
        output = res_block(output, 64, 3, 1, "block4")
        output = conv(output, 1, 1, 1, "conv2")
        output = tf.nn.tanh(output)
        refiner_vars = tf.contrib.framework.get_variables(scp)
    return output, refiner_vars


def discriminator(scope, input, reuse=False):
    with tf.variable_scope(scope, reuse=reuse) as scp:
        output = conv(input, 96, 3, 2, scope="conv1")
        output = conv(output, 64, 3, 2, scope="conv2")
        output = max_pool(output, 3, 1)
        output = conv(output, 32, 3, 1, scope="conv3")
        output = conv(output, 32, 1, 1, scope="conv4")
        logits = conv(output, 2, 1, 1, scope="conv5")
        output = tf.nn.softmax(logits, name="softmax")
        discriminator_vars = tf.contrib.framework.get_variables(scp)
    return output, logits, discriminator_vars


# Eliminating gradient explosion
def minimize(optimizer, loss, vars, max_grad_norm):
    grads_and_vars = optimizer.compute_gradients(loss)
    new_grads_and_vars = []
    for i, (grad, var) in enumerate(grads_and_vars):
        if grad is not None and var in vars:
            new_grads_and_vars.append((tf.clip_by_norm(grad, max_grad_norm), var))
    return optimizer.apply_gradients(new_grads_and_vars)


# Placeholder
R_input = tf.placeholder(tf.float32, [None, None, None, 1])
D_image = tf.placeholder(tf.float32, [None, None, None, 1])

# [None,None,None,1]->[-1,35,55,1]
r_input = tf.image.resize_images(R_input, [35, 55])
d_image = tf.image.resize_images(D_image, [35, 55])

# Network
R_output, refiner_vars = refiner("Refiner", r_input)
D_fake_output, D_fake_logits, discriminator_vars = discriminator("Discriminator", R_output)
D_real_output, D_real_logits, _ = discriminator("Discriminator", d_image, True)

# Refiner loss
self_regulation_loss = tf.reduce_sum(tf.abs(R_output - r_input), [1, 2, 3], name="self_regularization_loss", )
refine_loss = tf.reduce_sum(
    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=D_fake_logits,
                                                   labels=tf.ones_like(D_fake_logits, dtype=tf.int32)[:, :, :, 0]),
    [1, 2])
refiner_loss = tf.reduce_mean(0.5 * self_regulation_loss + refine_loss)

# Discriminator loss
discriminate_real_loss = tf.reduce_sum(
    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=D_real_logits,
                                                   labels=tf.ones_like(D_real_logits, dtype=tf.int32)[:, :, :, 0]),
    [1, 2])
discriminate_fake_loss = tf.reduce_sum(
    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=D_fake_logits,
                                                   labels=tf.zeros_like(D_fake_logits, dtype=tf.int32)[:, :, :, 0]),
    [1, 2])
discriminator_loss = tf.reduce_mean(discriminate_real_loss + discriminate_fake_loss)

# Training step
optimizer = tf.train.GradientDescentOptimizer(0.001)
sf_step = minimize(optimizer, self_regulation_loss, refiner_vars, 50)
refiner_step = minimize(optimizer, refiner_loss, refiner_vars, 50)
discriminator_step = minimize(optimizer, discriminator_loss, discriminator_vars, 50)

# Saver
saver = tf.train.Saver()

# Session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Summary
tf.summary.scalar("Refiner Loss", refiner_loss)
tf.summary.scalar("Discriminator Loss", discriminator_loss)
merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter("./graphs", sess.graph)

# Path setting
data = Data("./data/syn/", "./data/real/")
buffer = Buffer("./buffer/")
sample = Sample("./samples/")
syn_sample = data.syn_sample(1)
real_sample = data.real_sample(1)
syn_batch = data.syn_sample(16)

# Step 1
if not os.path.exists("./logs/step1/"):
    print("[*] Training starts.")
    for i in range(1000):

        mini_batch = data.syn_sample(32)
        sess.run(sf_step, feed_dict={R_input: mini_batch})
        buffer.push(sess.run(R_output, feed_dict={R_input: mini_batch}))

        print((i + 1) / 10, "%", "SRL:",
              sess.run(tf.reduce_mean(self_regulation_loss), feed_dict={R_input: syn_sample}))
        summary = sess.run(merged_summary, feed_dict={R_input: syn_sample})
        writer.add_summary(summary, global_step=i)

        if (i + 1) % 100 == 0:
            sample.push(sess.run(R_output, feed_dict={R_input: syn_sample}))

    print("[*] Step 1 finished. ")
    saver.save(sess, "./logs/step1/")
else:
    print("[*] Step 1 finished. ")
    saver.restore(sess, "./logs/step1/")

# Step 2
if not os.path.exists("./logs/step2/"):
    print("[*] Training starts.")
    for i in range(200):
        syn_batch = data.syn_sample(32)
        real_batch = data.real_sample(32)
        sess.run(discriminator_step, feed_dict={R_input: syn_batch, D_image: real_batch})
        print((i + 1) / 2, "%", "DL:",
              sess.run(discriminator_loss, feed_dict={R_input: syn_sample, D_image: real_sample}))
        summary = sess.run(merged_summary, feed_dict={R_input: syn_sample, D_image: real_sample})
        writer.add_summary(summary, global_step=i)

    print("[*] Step 2 finished. ")
    saver.save(sess, "./logs/step2/")

else:
    print("[*] Step 2 finished. ")
    saver.restore(sess, "./logs/step2/")

stuff_batch = data.syn_sample(100)
buffer.push(sess.run(R_output, feed_dict={R_input: stuff_batch}))

# Step 3
if not os.path.exists("./logs/step3/"):
    print("[*] Training starts.")
    for i in range(1000):

        for j in range(2):
            mini_batch = data.syn_sample(32)
            sess.run(refiner_step, feed_dict={R_input: mini_batch})
            print((i + 1) / 100.0, "%", "RL:", sess.run(refiner_loss, feed_dict={R_input: syn_sample}))

        new_syn_sample = data.syn_sample(16)
        new_refined_batch = sess.run(R_output, feed_dict={R_input: new_syn_sample})
        history_batch = buffer.sample(16)
        concat_batch = np.concatenate([new_refined_batch, history_batch], axis=0)

        for k in range(1):
            real_batch = data.real_sample(32)
            sess.run(discriminator_step, feed_dict={R_input: concat_batch, D_image: real_batch})
            print((i + 1) / 100.0, "%", "DL:",
                  sess.run(discriminator_loss, feed_dict={R_input: syn_sample, D_image: real_sample}))

        buffer.random_replace(new_refined_batch)

        summary = sess.run(merged_summary, feed_dict={R_input: syn_sample, D_image: real_sample})
        writer.add_summary(summary, global_step=i)

        sample_batch = sess.run(R_output, feed_dict={R_input: syn_batch})
        sample.push(concat(sample_batch))


    print("[*] Step 3 finished. ")
    saver.save(sess, "./logs/step3/")
else:
    saver.restore(sess, "./logs/step3/")

print("[*] Model is ready to use.")
