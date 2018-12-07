import tensorflow as tf

from vae_carl.helpers import batch_index_groups, dtype
from vae_carl import mnist_data
import vae_carl.vae as vae

train_total_data, _, _, _, test_data, test_labels = mnist_data.prepare_MNIST_data()

train_size = 10000
IMAGE_SIZE_MNIST = 28
num_hidden = 500
dim_img = IMAGE_SIZE_MNIST ** 2
dim_z = 2
learn_rate = 1e-3
batch_size = min(128, train_size)
num_epochs = 10

y_input = tf.placeholder(dtype, shape=[None, dim_img], name='input_img')
y_output_true = tf.placeholder(dtype, shape=[None, dim_img], name='target_img')

# dropout
keep_prob = tf.placeholder(dtype, name='keep_prob')

# network architecture
ae = vae.autoencoder(
    y_input=y_input,
    y_output_true=y_output_true,
    dim_img=dim_img,
    dim_z=dim_z,
    num_hidden=num_hidden,
    keep_prob=keep_prob
)

# optimization
train_step = tf.train.AdamOptimizer(learn_rate).minimize(ae.loss)

y_train = train_total_data[:train_size, :-mnist_data.NUM_LABELS]
y_train_labels = train_total_data[:train_size, -mnist_data.NUM_LABELS:]

print("Num data points", train_size)
print("Num epochs", num_epochs)

with tf.Session() as session:

    session.run(tf.global_variables_initializer())
    session.graph.finalize()

    for epoch in range(num_epochs):
        for i, batch_indices in enumerate(batch_index_groups(batch_size=batch_size, num_samples=train_size)):

            batch_xs_input = y_train[batch_indices, :]

            _, tot_loss, loss_likelihood, loss_divergence = session.run(
                (
                    train_step,
                    ae.loss,
                    ae.neg_marginal_likelihood,
                    ae.kl_divergence
                ),
                feed_dict={
                    y_input: batch_xs_input,
                    y_output_true: batch_xs_input,
                    keep_prob: 0.9
                }
            )

        print(
            "epoch %d: L_tot %03.2f L_likelihood %03.2f L_divergence %03.2f" % (
                epoch,
                tot_loss,
                loss_likelihood,
                loss_divergence
            )
        )