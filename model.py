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
        output = conv(output, 3, 1, 1, "conv2")
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
R_input = tf.placeholder(tf.float32, [None, 35, 55, 3])
D_image = tf.placeholder(tf.float32, [None, 35, 55, 3])
dB = tf.placeholder(tf.float32)

# Network
R_output, refiner_vars = refiner("Refiner", R_input)
D_fake_output, D_fake_logits, discriminator_vars = discriminator("Discriminator", R_output)
D_real_output, D_real_logits, _ = discriminator("Discriminator", D_image, True)

# Refiner loss
self_regulation_loss = tf.reduce_sum(tf.abs(R_output - R_input), [1, 2, 3], name="self_regularization_loss")
refine_loss = tf.reduce_sum(
    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=D_fake_logits,
                                                   labels=tf.ones_like(D_fake_logits, dtype=tf.int32)[:, :, :, 0]),
    [1, 2])
refiner_loss = tf.reduce_mean(3.0 * self_regulation_loss + refine_loss)

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
optimizer = tf.train.AdamOptimizer(0.001)
sf_step = minimize(optimizer, self_regulation_loss, refiner_vars, 50)
refiner_step = minimize(optimizer, refiner_loss, refiner_vars, 50)
discriminator_step = minimize(optimizer, discriminator_loss, discriminator_vars, 50)

# Saver
saver = tf.train.Saver()

# Session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Summary
tf.summary.scalar("PSNR", dB)
tf.summary.scalar("Self Regularization Loss", tf.reduce_mean(self_regulation_loss))
tf.summary.scalar("Refine Loss", tf.reduce_mean(refine_loss))
tf.summary.scalar("Discriminator Loss on current refined image", discriminator_loss)
merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter("./graphs", sess.graph)

# Path setting
data = Data()
buffer = Buffer()
sample = Sample()
r_sample = data.r_sample(1)
n_sample = data.n_sample(1)
r_batch = data.r_sample(16)
sample.push(concat(r_batch))

# --------------------------------------------------------------------------------------------------------------------
'''
# Step 1
if not os.path.exists("./logs/step1/"):
    print("[*] Training in progress...")
    for i in range(1000):
        # Train SR
        rain_batch = data.r_sample(32)
        sess.run(sf_step, feed_dict={R_input: rain_batch})

        # Summary
        summary = sess.run(merged_summary, feed_dict={R_input: r_sample, D_image: n_sample})
        writer.add_summary(summary, global_step=i)

        # Update buffer
        buffer_batch = data.r_sample(2)
        buffer.push(sess.run(R_output, feed_dict={R_input: buffer_batch}))

        # Sample
        if (i + 1) % 100 == 0:
            sample.push(sess.run(R_output, feed_dict={R_input: r_sample}))

    saver.save(sess, "./logs/step1/")
else:
    saver.restore(sess, "./logs/step1/")

print("[*] Step 1 finished. ")

print("SRL:", sess.run(tf.reduce_mean(self_regulation_loss), feed_dict={R_input: r_sample}))

# Step 2
if not os.path.exists("./logs/step2/"):
    print("[*] Training in progress...")
    for i in range(200):
        # Concat with history
        new_r_sample = data.r_sample(16)
        new_refined_batch = sess.run(R_output, feed_dict={R_input: new_r_sample})
        history_batch = buffer.sample(16)
        concat_batch = np.concatenate([new_refined_batch, history_batch], axis=0)
        np.random.shuffle(concat_batch)

        # Train D
        no_batch = data.n_sample(32)
        sess.run(discriminator_step, feed_dict={R_input: concat_batch, D_image: no_batch})

        # Summary
        summary = sess.run(merged_summary, feed_dict={R_input: r_sample, D_image: n_sample})
        writer.add_summary(summary, global_step=i)

    saver.save(sess, "./logs/step2/")
else:
    saver.restore(sess, "./logs/step2/")

print("[*] Step 2 finished. ")

print("DL:", sess.run(discriminator_loss, feed_dict={R_input: r_sample, D_image: n_sample}))
'''

# Step 3
if not os.path.exists("./logs/step3/"):
    print("[*] Training in progress...")
    for i in range(5000):

        # Train R
        for j in range(1):
            mini_batch = data.r_sample(32)
            sess.run(refiner_step, feed_dict={R_input: mini_batch})

        # Update buffer
        new_r_sample = data.r_sample(16)
        new_refined_batch = sess.run(R_output, feed_dict={R_input: new_r_sample})
        if i < 100:
            buffer.push(new_refined_batch)
        else:
            buffer.random_replace(new_refined_batch)

        # Train D
        for k in range(2):
            # Concat with history
            new_r_sample = data.r_sample(16)
            new_refined_batch = sess.run(R_output, feed_dict={R_input: new_r_sample})
            history_batch = buffer.sample(16)
            concat_batch = np.concatenate([new_refined_batch, history_batch], axis=0)
            np.random.shuffle(concat_batch)
            # Train D
            mini_batch = data.n_sample(32)
            sess.run(discriminator_step, feed_dict={R_input: concat_batch, D_image: mini_batch})

        # PSNR
        matrix1 = normalize(read_image("./data/r/" + "30.png"))  # 14.415dB
        matrix1 = sess.run(R_output, feed_dict={R_input: matrix1})
        matrix2 = normalize(read_image("./data/n/" + "30.png"))
        psnr = PSNR(matrix1, matrix2)

        # Summary
        summary = sess.run(merged_summary, feed_dict={R_input: r_sample, D_image: n_sample, dB: psnr})
        writer.add_summary(summary, global_step=i)

        if (i + 1) % 50 == 0:
            # Sample
            sample_batch = sess.run(R_output, feed_dict={R_input: r_batch})
            sample.push(concat(sample_batch))

    saver.save(sess, "./logs/step3/")
else:
    saver.restore(sess, "./logs/step3/")

print("[*] Step 3 finished. ")
