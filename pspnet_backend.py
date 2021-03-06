from __future__ import absolute_import
import os
import numpy as np
import cv2
from tqdm import tqdm
import tensorflow as tf
from .model import PSPNet101, PSPNet50
from .tools import prepare_label, decode_labels

from protoseg.backends import AbstractBackend
from protoseg.trainer import Trainer

from tensorboardX import SummaryWriter


class pspnet_backend(AbstractBackend):

    config = tf.ConfigProto(
        device_count={'GPU': 0}
    )
    saver = None
    sess = None
    step = 0
    x = None
    y = None

    def __init__(self):
        AbstractBackend.__init__(self)

    def load_model(self, config, modelfile):
        with tf.name_scope("create_inputs"):
            self.x = tf.placeholder(tf.float32, shape=(
                None, config['width'], config['height'], 3), name='batch')
            self.y = tf.placeholder(tf.uint8, shape=(
                None, config['width'], config['height'], 1), name='batch')

        if config['backbone'] == 'pspnet101':
            model = PSPNet101({'data': self.x}, is_training=True,
                              num_classes=config['classes'])
        else:
            model = PSPNet50({'data': self.x}, is_training=True,
                             num_classes=config['classes'])
        #if os.path.isfile(modelfile):
        #    print('loaded model from:', modelfile)
        
        return model

    def save_model(self, model):
        self.saver.save(self.sess, model.modelfile, global_step=self.step)
        print('saved model to:', model.modelfile)

    def init_trainer(self, trainer):
        net = trainer.model.model
        config = trainer.config
        raw_output = net.layers['conv6']

        # According from the prototxt in Caffe implement, learning rate must multiply by 10.0 in pyramid module
        fc_list = ['conv5_3_pool1_conv', 'conv5_3_pool2_conv',
                   'conv5_3_pool3_conv', 'conv5_3_pool6_conv', 'conv6', 'conv5_4']
        restore_var = [v for v in tf.global_variables()]
        all_trainable = [v for v in tf.trainable_variables() if (
            'beta' not in v.name and 'gamma' not in v.name) or config.get('beta_gamma')]
        fc_trainable = [
            v for v in all_trainable if v.name.split('/')[0] in fc_list]
        conv_trainable = [v for v in all_trainable if v.name.split(
            '/')[0] not in fc_list]  # lr * 1.0
        fc_w_trainable = [
            v for v in fc_trainable if 'weights' in v.name]  # lr * 10.0
        fc_b_trainable = [
            v for v in fc_trainable if 'biases' in v.name]  # lr * 20.0
        assert(len(all_trainable) == len(
            fc_trainable) + len(conv_trainable))
        assert(len(fc_trainable) == len(
            fc_w_trainable) + len(fc_b_trainable))

        # Predictions: ignoring all predictions with labels greater or equal than n_classes
        raw_prediction = tf.reshape(raw_output, [-1, config['classes']])
        label_proc = prepare_label(self.y, tf.stack(raw_output.get_shape()[
                                   1:3]), num_classes=config['classes'], one_hot=False)
        raw_gt = tf.reshape(label_proc, [-1, ])
        indices = tf.squeeze(
            tf.where(tf.less_equal(raw_gt, config['classes'] - 1)), 1)
        gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
        prediction = tf.gather(raw_prediction, indices)

        # Predictions.
        img_shape = [config['width'], config['height'], 3]
        raw_output_up = tf.image.resize_bilinear(
            raw_output, size=[config['height'], config['width']], align_corners=True)
        raw_output_up = tf.image.crop_to_bounding_box(
            raw_output_up, 0, 0, img_shape[0], img_shape[1])
        raw_output_up = tf.argmax(raw_output_up, axis=3)
        self.pred = tf.expand_dims(raw_output_up, dim=3)

        # Pixel-wise softmax loss.
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=prediction, labels=gt)
        l2_losses = [
            0.0001 * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]
        trainer.reduced_loss = tf.reduce_mean(loss) + tf.add_n(l2_losses)

        # Using Poly learning rate policy
        base_lr = tf.constant(config['learn_rate'])
        trainer.step_ph = tf.placeholder(dtype=tf.float32, shape=())
        learning_rate = tf.scalar_mul(base_lr, tf.pow(
            (1 - trainer.step_ph / len(trainer.dataloader)), 0.9))

        if True:
            update_ops = None
        else:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            opt_conv = tf.train.MomentumOptimizer(learning_rate, 0.9)
            opt_fc_w = tf.train.MomentumOptimizer(learning_rate * 10.0, 0.9)
            opt_fc_b = tf.train.MomentumOptimizer(learning_rate * 20.0, 0.9)

            grads = tf.gradients(
                trainer.reduced_loss, conv_trainable + fc_w_trainable + fc_b_trainable)
            grads_conv = grads[:len(conv_trainable)]
            grads_fc_w = grads[len(conv_trainable): (
                len(conv_trainable) + len(fc_w_trainable))]
            grads_fc_b = grads[(len(conv_trainable) + len(fc_w_trainable)):]

            train_op_conv = opt_conv.apply_gradients(
                zip(grads_conv, conv_trainable))
            train_op_fc_w = opt_fc_w.apply_gradients(
                zip(grads_fc_w, fc_w_trainable))
            train_op_fc_b = opt_fc_b.apply_gradients(
                zip(grads_fc_b, fc_b_trainable))

            trainer.train_op = tf.group(
                train_op_conv, train_op_fc_w, train_op_fc_b)

        self.init = tf.global_variables_initializer()
        self.config = tf.ConfigProto(
            device_count={'GPU': config['gpu']}
        )
        self.config.gpu_options.allow_growth = True
        self.config.allow_soft_placement = True
        self.config.gpu_options.allocator_type = 'BFC'
        self.sess = tf.Session(config=self.config)
        self.sess.run(self.init)
        self.saver = tf.train.Saver(
            var_list=tf.global_variables(), max_to_keep=2)
        modelpath = os.path.dirname(trainer.model.modelfile)
        ckpt = tf.train.get_checkpoint_state(modelpath)
        if ckpt and ckpt.model_checkpoint_path:
            loader = tf.train.Saver(var_list=restore_var)
            load_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
            loader.restore(self.sess, ckpt.model_checkpoint_path)
            print("Restored model parameters from {}".format(ckpt.model_checkpoint_path))
        else:
            print('No checkpoint file found.')
            load_step = 0

    def dataloader_format(self, img, mask=None):
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
        img = img.astype(np.float32)
        img = img/255.0
        if mask is None:
            return tf.convert_to_tensor(img)

        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        mask[mask > 0] = 1  # binary mask
        mask = np.expand_dims(mask, axis=2)
        mask = mask.astype(np.uint8)
        return img, mask

    def train_epoch(self, trainer):
        print('train on pspnet backend')
        batch_size = trainer.config['batch_size']
        summarysteps = trainer.config['summarysteps']

        for i, (img_batch, label_batch) in tqdm(enumerate(trainer.dataloader.batch_generator(batch_size))):
            x_batch = np.array(img_batch)
            y_batch = np.array(label_batch)
            feed_dict = {trainer.step_ph: i,
                         self.x: x_batch, self.y: y_batch}
            loss_value, _ = self.sess.run(
                [trainer.reduced_loss, trainer.train_op], feed_dict=feed_dict)
            trainer.loss += loss_value
            if i % summarysteps == 0:
                print(trainer.global_step, i, 'loss:', trainer.loss / (i+1))
                if trainer.summarywriter:
                    trainer.summarywriter.add_scalar(
                        trainer.name+'loss', loss_value, global_step=trainer.global_step)
                    img = np.transpose(x_batch[0], axes=[2, 0, 1])
                    trainer.summarywriter.add_image(
                        trainer.name+'image', img, global_step=trainer.global_step)
                    trainer.summarywriter.add_image(
                        trainer.name+'mask', np.squeeze(y_batch[0], axis=2).astype(np.float32), global_step=trainer.global_step)
                    pred = self.predict(None, x_batch[0])
                    trainer.summarywriter.add_image(
                        trainer.name+'predicted', np.squeeze(pred, axis=2).astype(np.float32), global_step=trainer.global_step)
            trainer.global_step += 1
                    

    def validate_epoch(self, trainer):
        batch_size = trainer.config['batch_size']
        for i, (img_batch, label_batch) in enumerate(trainer.valdataloader.batch_generator(batch_size)):
            x_batch = np.array(img_batch)
            y_batch = np.array(label_batch)
            prediction = self.batch_predict(trainer, x_batch)
            trainer.metric(
                np.squeeze(prediction[0], axis=2), np.squeeze(y_batch[0], axis=2), prefix=trainer.name)
            if trainer.summarywriter:
                img = np.transpose(x_batch[0], axes=[2, 0, 1])
                trainer.summarywriter.add_image(
                    trainer.name+"val_image", img, global_step=trainer.epoch)
                trainer.summarywriter.add_image(
                    trainer.name+"val_mask", np.squeeze(y_batch[0], axis=2).astype(np.float32), global_step=trainer.epoch)
                trainer.summarywriter.add_image(
                    trainer.name+"val_predicted", np.squeeze(prediction[0], axis=2).astype(np.float32), global_step=trainer.epoch)

    def get_summary_writer(self, logdir='results/'):
        return SummaryWriter(log_dir=logdir)

    def predict(self, predictor, img):
        img_batch = [img]
        return self.batch_predict(predictor, img_batch)[0]

    def batch_predict(self, predictor, img_batch):
        feed_dict = {self.x: img_batch}
        preds = self.sess.run(self.pred, feed_dict=feed_dict)
        return preds
