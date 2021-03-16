import os
import time
import cv2
import vgg19
import numpy as np
import tensorflow as tf
import os.path as ops
import math
# from skimage.measure import compare_psnr
# from skimage.measure import compare_ssim
# from skimage.measure import compare_mse
# from skimage.measure import compare_nrmse
from operations import TransposeConv, DropOut
from operations import Conv, ReLU, LeakyReLU, AvgPool, BatchNorm
from PIL import Image
from numpy import average, linalg, dot

# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
# os.environ['CUDA_VISIBLE_DEVICES'] = '/device:XLA_CPU:0'
tf.compat.v1.disable_eager_execution()


class GAN():

    def __init__(self, args):
        self.num_discriminator_filters = args.D_filters
        self.layers = args.layers
        self.growth_rate = args.growth_rate
        self.gan_wt = args.gan_wt
        self.l1_wt = args.l1_wt
        self.vgg_wt = args.vgg_wt
        self.restore = args.restore
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.lr = args.lr
        self.model_name = args.model_name
        self.decay = args.decay
        self.save_samples = args.save_samples
        self.sample_image_dir = args.sample_image_dir
        self.A_dir = args.A_dir
        self.B_dir = args.B_dir
        self.custom_data = args.custom_data
        self.val_fraction = args.val_fraction
        self.val_threshold = args.val_threshold
        self.val_frequency = args.val_frequency
        self.logger_frequency = args.logger_frequency

        self.EPS = 10e-12
        self.score_best = -1
        self.ckpt_dir = os.path.join(os.getcwd(), self.model_name, 'checkpoint')
        # self.ckpt_dir = args.weights_path
        self.tensorboard_dir = os.path.join(os.getcwd(), self.model_name, 'tensorboard')
        # self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')

    def Layer(self, input_):
        """
        This function creates the components inside a composite layer
        of a Dense Block.
        """
        with tf.compat.v1.variable_scope("Composite"):
            next_layer = BatchNorm(input_, isTrain=self.isTrain)
            next_layer = ReLU(next_layer)
            next_layer = Conv(next_layer, kernel_size=3, stride=1, output_channels=self.growth_rate)
            next_layer = DropOut(next_layer, isTrain=self.isTrain, rate=0.2)

            return next_layer

    def TransitionDown(self, input_, name):

        with tf.compat.v1.variable_scope(name):
            reduction = 0.5
            reduced_output_size = int(int(input_.get_shape()[-1]) * reduction)

            next_layer = BatchNorm(input_, isTrain=self.isTrain)
            next_layer = Conv(next_layer, kernel_size=1, stride=1, output_channels=reduced_output_size)
            next_layer = DropOut(next_layer, isTrain=self.isTrain, rate=0.2)
            next_layer = AvgPool(next_layer)

            return next_layer

    def TransitionUp(self, input_, output_channels, name):

        with tf.compat.v1.variable_scope(name):
            next_layer = TransposeConv(input_, output_channels=output_channels, kernel_size=3)

            return next_layer

    def DenseBlock(self, input_, name, layers=4):

        with tf.compat.v1.variable_scope(name):
            for i in range(layers):
                with tf.compat.v1.variable_scope("Layer" + str(i + 1)) as scope:
                    output = self.Layer(input_)
                    output = tf.concat([input_, output], axis=3)
                    input_ = output

        return output

    def generator(self, input_, name):
        with tf.compat.v1.variable_scope(name, reuse=False):
            """
            54 Layer Tiramisu
            """
            with tf.compat.v1.variable_scope('InputConv') as scope:
                input_ = Conv(input_, kernel_size=3, stride=1, output_channels=self.growth_rate * 4)

            collect_conv = []

            for i in range(1, 6):
                input_ = self.DenseBlock(input_, name='Encoder' + str(i), layers=self.layers)
                collect_conv.append(input_)
                input_ = self.TransitionDown(input_, name='TD' + str(i))

            input_ = self.DenseBlock(input_, name='BottleNeck', layers=15)

            for i in range(1, 6):
                input_ = self.TransitionUp(input_, output_channels=self.growth_rate * 4, name='TU' + str(6 - i))
                input_ = tf.concat([input_, collect_conv[6 - i - 1]], axis=3, name='Decoder' + str(6 - i) + '/Concat')
                input_ = self.DenseBlock(input_, name='Decoder' + str(6 - i), layers=self.layers)

            with tf.compat.v1.variable_scope('OutputConv') as scope:
                output = Conv(input_, kernel_size=1, stride=1, output_channels=3)

            return tf.nn.tanh(output)

    def generator_i(self, input_):
        with tf.compat.v1.variable_scope("NEW", reuse=tf.compat.v1.AUTO_REUSE):
            """
            54 Layer Tiramisu
            """
            with tf.compat.v1.variable_scope('InputConv') as scope:
                input_ = Conv(input_, kernel_size=3, stride=1, output_channels=self.growth_rate * 4)

            collect_conv = []

            for i in range(1, 6):
                input_ = self.DenseBlock(input_, name='Encoder' + str(i), layers=self.layers)
                collect_conv.append(input_)
                input_ = self.TransitionDown(input_, name='TD' + str(i))

            input_ = self.DenseBlock(input_, name='BottleNeck', layers=15)

            for i in range(1, 6):
                input_ = self.TransitionUp(input_, output_channels=self.growth_rate * 4, name='TU' + str(6 - i))
                input_ = tf.concat([input_, collect_conv[6 - i - 1]], axis=3, name='Decoder' + str(6 - i) + '/Concat')
                input_ = self.DenseBlock(input_, name='Decoder' + str(6 - i), layers=self.layers)

            with tf.compat.v1.variable_scope('OutputConv') as scope:
                output = Conv(input_, kernel_size=1, stride=1, output_channels=3)

            return tf.nn.tanh(output)

    def discriminator(self, input_, target, stride=2, layer_count=4):
        with tf.compat.v1.variable_scope("DIS", reuse=tf.compat.v1.AUTO_REUSE):
            """
            Using the PatchGAN as a discriminator
            """
            input_ = tf.concat([input_, target], axis=3, name='Concat')
            layer_specs = self.num_discriminator_filters * np.array([1, 2, 4, 8])

            for i, output_channels in enumerate(layer_specs, 1):

                with tf.compat.v1.variable_scope('Layer' + str(i)) as scope:

                    if i != 1:
                        input_ = BatchNorm(input_, isTrain=self.isTrain)

                    if i == layer_count:
                        stride = 1

                    input_ = LeakyReLU(input_)
                    input_ = Conv(input_, output_channels=output_channels, kernel_size=4, stride=stride,
                                  padding='VALID',
                                  mode='discriminator')

            with tf.compat.v1.variable_scope('Final_Layer') as scope:
                output = Conv(input_, output_channels=1, kernel_size=4, stride=1, padding='VALID', mode='discriminator')

            return tf.sigmoid(output)

    def build_vgg(self, img):

        model = vgg19.Vgg19()
        img = tf.compat.v1.image.resize_images(img, [224, 224])
        layer = model.feature_map(img)
        return layer

    def build_model(self):

        with tf.compat.v1.variable_scope('Placeholders') as scope:
            self.RealA = tf.compat.v1.placeholder(name='A', shape=[None, 256, 256, 3], dtype=tf.float32)
            self.RealB = tf.compat.v1.placeholder(name='B', shape=[None, 256, 256, 3], dtype=tf.float32)
            self.isTrain = tf.compat.v1.placeholder(name="isTrain", shape=None, dtype=tf.bool)
            self.step = tf.compat.v1.train.get_or_create_global_step()

        with tf.compat.v1.variable_scope('Generator', reuse=tf.compat.v1.AUTO_REUSE) as scope:
            # Coarse Removal
            self.Fake_coarse = self.generator(self.RealA, "COARSE")  # 粗糙去阴影图 "COARSE"参数
            # Residual
            self.GT_residual = self.RealB - self.RealA  # 真实残差输出
            self.Fake_residual = self.generator(self.RealA, "RES")  # 残差gen,虚假残差输出 "RES"参数
            self.Fake_C_res = self.Fake_residual + self.RealA  # 虚假残差->去阴影图
            # Illumination
            self.GT_illum = tf.divide(self.RealA, self.RealB)  # 真实逆光照输出
            self.Fake_illum = self.generator(self.RealA, "ILLUM")  # 虚假逆光照输出 "NEW"参数
            self.Fake_C_illum = tf.divide(self.RealA, self.Fake_illum)  # 逆光照->去阴影图
            # Refined Removal
            self.Input_concat = tf.concat([self.Fake_coarse, self.Fake_C_res], axis=3)
            self.Input_concat = tf.concat([self.Input_concat, self.Fake_C_illum], axis=3)

            self.Fake_final = self.generator(self.Input_concat, "REFINED")

        with tf.name_scope('Real_Discriminator'):
            with tf.compat.v1.variable_scope('Discriminator', reuse=tf.compat.v1.AUTO_REUSE) as scope:
                self.predict_real_final = self.discriminator(self.RealA, self.RealB)

        with tf.name_scope('Fake_Discriminator'):
            with tf.compat.v1.variable_scope('Discriminator', reuse=True) as scope:
                self.predict_fake_final = self.discriminator(self.RealA, self.Fake_final)
                self.predict_fake_res = self.discriminator(self.RealA, self.Fake_C_res)
                self.predict_fake_illum = self.discriminator(self.RealA, self.Fake_C_illum)

        with tf.name_scope('Real_VGG'):
            with tf.compat.v1.variable_scope('VGG') as scope:
                self.RealB_VGG = self.build_vgg(self.RealB)

        with tf.name_scope('Fake_VGG'):
            with tf.compat.v1.variable_scope('VGG', reuse=True) as scope:
                self.Fake_coarse_VGG = self.build_vgg(self.Fake_coarse)
                self.Fake_final_VGG = self.build_vgg(self.Fake_final)

        with tf.name_scope('DiscriminatorLoss'):
            self.D_loss_final = tf.reduce_mean(
                -(tf.compat.v1.log(self.predict_real_final + self.EPS) + tf.compat.v1.log(
                    1 - self.predict_fake_final + self.EPS)))
            self.D_loss_res = tf.reduce_mean(- tf.compat.v1.log(1 - self.predict_fake_res + self.EPS))
            self.D_loss_illum = tf.reduce_mean(- tf.compat.v1.log(1 - self.predict_fake_illum + self.EPS))
            self.D_loss = self.D_loss_final + self.D_loss_res + self.D_loss_illum

        with tf.name_scope('GeneratorLoss'):
            # L_rem
            self.L_vis = tf.reduce_mean(
                tf.abs(self.RealB - self.Fake_coarse) + tf.abs(self.RealB - self.Fake_final))  # L_vis
            self.L_percept = 1e-3 * tf.reduce_mean(  # L_percept
                tf.losses.mean_squared_error(self.RealB_VGG, self.Fake_coarse_VGG) + tf.losses.mean_squared_error(
                    self.RealB_VGG, self.Fake_final_VGG))
            self.L_rem = self.L_vis + 0.1 * self.L_percept
            # L_res
            self.L_res = tf.reduce_mean(tf.abs(self.GT_residual - self.Fake_residual))
            # L_illum
            # self.L_illum = tf.reduce_mean(tf.clip_by_value(tf.abs(self.GT_illum - self.Fake_illum), -2., 2.))
            self.L_illum = tf.reduce_mean(tf.clip_by_value(tf.abs(self.GT_illum - self.Fake_illum), -1e5, 1e5))
            # L_cross
            self.L_cross_res = tf.reduce_mean(tf.abs(self.RealB - self.Fake_C_res))  # L_cross_res
            self.L_cross_illum = tf.reduce_mean(tf.abs(self.RealB - self.Fake_C_illum))  # L_cross_illum
            self.L_cross = self.L_cross_res + 0.2 * self.L_cross_illum
            # L_adv
            self.G_loss_final = tf.reduce_mean(-(tf.compat.v1.log(self.predict_fake_final + self.EPS)))
            self.G_loss_res = tf.reduce_mean(-tf.compat.v1.log(self.predict_fake_res + self.EPS))
            self.G_loss_illum = tf.reduce_mean(-tf.compat.v1.log(self.predict_fake_illum + self.EPS))
            self.L_adv = self.G_loss_final + self.G_loss_res + self.G_loss_illum
            # total G_loss
            self.G_loss = 10 * self.L_res + 100 * self.L_rem + 50 * self.L_illum + self.L_cross + self.L_adv

        # with tf.name_scope('Summary'):
        #     D_loss_sum = tf.compat.v1.summary.scalar('Discriminator Loss', self.D_loss)
        #     G_loss_sum = tf.compat.v1.summary.scalar('Generator Loss', self.G_loss)
        #     # gan_loss_sum = tf.summary.scalar('GAN Loss', self.gan_loss)
        #     # l1_loss_sum = tf.summary.scalar('L1 Loss', self.l1_loss)
        #     # l1_e_loss_sum = tf.summary.scalar('Ang Loss', self.l1_e_loss)
        #     # e_loss_sum = tf.summary.scalar('Ang Loss', self.e_loss)
        #     # vgg_loss_sum = tf.summary.scalar('VGG Loss', self.gan_loss)
        #     L_rem_sum = tf.compat.v1.summary.scalar('L_rem', self.L_rem)
        #     L_res_sum = tf.compat.v1.summary.scalar('L_res', self.L_res)
        #     L_illum_sum = tf.compat.v1.summary.scalar('L_illum', self.L_illum)
        #     L_cross_sum = tf.compat.v1.summary.scalar('L_cross', self.L_cross)
        #     L_adv_sum = tf.compat.v1.summary.scalar('L_adv', self.L_adv)
        #     output_img = tf.compat.v1.summary.image('Output', self.Fake_final, max_outputs=1)
        #     target_img = tf.compat.v1.summary.image('Target', self.RealB, max_outputs=1)
        #     input_img = tf.compat.v1.summary.image('Input', self.RealA, max_outputs=1)
        #
        #     self.image_summary = tf.compat.v1.summary.merge([output_img, target_img, input_img])
        #     # self.G_summary = tf.summary.merge(
        #     #     [gan_loss_sum, l1_loss_sum, l1_e_loss_sum, e_loss_sum, vgg_loss_sum, G_loss_sum])
        #     # self.D_summary = D_loss_sum
        #     self.G_summary = tf.compat.v1.summary.merge(
        #         [L_rem_sum, L_res_sum, L_illum_sum, L_cross_sum, L_adv_sum, G_loss_sum])
        #     self.D_summary = D_loss_sum

        with tf.name_scope('Variables'):
            self.G_vars = [var for var in tf.compat.v1.trainable_variables() if var.name.startswith("Generator")]
            self.D_vars = [var for var in tf.compat.v1.trainable_variables() if var.name.startswith("Discriminator")]

        with tf.name_scope('Save'):
            self.saver = tf.compat.v1.train.Saver(max_to_keep=3)

        with tf.name_scope('Optimizer'):
            update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                with tf.name_scope("Discriminator_Train"):
                    D_optimizer = tf.compat.v1.train.AdamOptimizer(self.lr, beta1=0.5)
                    self.D_grads_and_vars = D_optimizer.compute_gradients(self.D_loss, var_list=self.D_vars)
                    self.D_train = D_optimizer.apply_gradients(self.D_grads_and_vars, global_step=self.step)

                with tf.name_scope("Generator_Train"):
                    G_optimizer = tf.compat.v1.train.AdamOptimizer(self.lr, beta1=0.5)
                    self.G_grads_and_vars = G_optimizer.compute_gradients(self.G_loss, var_list=self.G_vars)
                    self.G_train = G_optimizer.apply_gradients(self.G_grads_and_vars, global_step=self.step)

    def train(self):

        start_epoch = 0
        logger_frequency = self.logger_frequency
        val_frequency = self.val_frequency
        val_threshold = self.val_threshold

        if not os.path.exists(self.model_name):
            os.mkdir(self.model_name)

        print('Loading Model')
        self.build_model()
        print('Model Loaded')

        print('Loading Data')

        if self.custom_data:

            # Please ensure that the input images and target images have
            # the same filename.

            data = sorted(os.listdir(self.A_dir))

            total_image_count = int(len(data) * (1 - self.val_fraction))
            batches = total_image_count // self.batch_size

            train_data = data[: total_image_count]
            val_data = data[total_image_count:]
            val_image_count = len(val_data)

            self.A_train = np.zeros((total_image_count, 256, 256, 3))
            self.B_train = np.zeros((total_image_count, 256, 256, 3))
            self.A_val = np.zeros((val_image_count, 256, 256, 3))
            self.B_val = np.zeros((val_image_count, 256, 256, 3))

            print(self.A_train.shape, self.A_val.shape)

            for i, file in enumerate(train_data):
                self.A_train[i] = cv2.resize(
                    cv2.imread(os.path.join(os.getcwd(), self.A_dir, file), 1).astype(np.float32), (256, 256))
                self.B_train[i] = cv2.resize(
                    cv2.imread(os.path.join(os.getcwd(), self.B_dir, file), 1).astype(np.float32), (256, 256))

            for i, file in enumerate(val_data):
                self.A_val[i] = cv2.resize(
                    cv2.imread(os.path.join(os.getcwd(), self.A_dir, file), 1).astype(np.float32), (256, 256))
                self.B_val[i] = cv2.resize(
                    cv2.imread(os.path.join(os.getcwd(), self.B_dir, file), 1).astype(np.float32), (256, 256))

        else:

            self.A_train = np.load('A_train.npy').astype(np.float32)
            self.B_train = np.load('B_train.npy').astype(np.float32)
            self.A_val = np.load('A_val.npy').astype(np.float32)  # Valset 2
            self.B_val = np.load('B_val.npy').astype(np.float32)

            total_image_count = len(self.A_train)
            val_image_count = len(self.A_val)
            batches = total_image_count // self.batch_size

        self.A_val = (self.A_val / 255) * 2 - 1
        self.B_val = (self.B_val / 255) * 2 - 1
        self.A_train = (self.A_train / 255) * 2 - 1
        self.B_train = (self.B_train / 255) * 2 - 1

        print('Data Loaded')
        print("Total Batch: ", batches)
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # sess = tf.Session(config=config)

        # with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True)) as self.sess:
        with tf.compat.v1.Session() as self.sess:
            init_op = tf.compat.v1.global_variables_initializer()
            self.sess.run(init_op)

            if self.restore:
                print('Loading Checkpoint')
                ckpt = tf.train.latest_checkpoint(self.ckpt_dir)
                self.saver.restore(self.sess, ckpt)
                self.step = tf.compat.v1.train.get_or_create_global_step()
                print('Checkpoint Loaded')

            # self.writer = tf.compat.v1.summary.FileWriter(self.tensorboard_dir, tf.compat.v1.get_default_graph())
            total_parameter_count = tf.reduce_sum(
                [tf.reduce_prod(tf.shape(v)) for v in tf.compat.v1.trainable_variables()])
            G_parameter_count = tf.reduce_sum(
                [tf.reduce_prod(tf.shape(v)) for v in tf.compat.v1.trainable_variables() if
                 v.name.startswith("Generator")])
            D_parameter_count = tf.reduce_sum(
                [tf.reduce_prod(tf.shape(v)) for v in tf.compat.v1.trainable_variables() if
                 v.name.startswith("Discriminator")])
            loss_operations = [self.D_loss, self.G_loss, self.L_rem, self.L_vis, self.L_percept, self.L_res,
                               self.L_illum, self.L_cross, self.L_adv]

            counts = self.sess.run([G_parameter_count, D_parameter_count, total_parameter_count])

            print('Generator parameter count:', counts[0])
            print('Discriminator parameter count:', counts[1])
            print('Total parameter count:', counts[2])

            # The variable below is divided by 2 since both the Generator 
            # and the Discriminator increases step count by 1
            start = self.step.eval() // (batches * 2)

            for i in range(start, self.epochs):

                print('Epoch:', i)
                shuffle = np.random.permutation(total_image_count)

                for j in range(batches):

                    if j != batches - 1:
                        current_batch = shuffle[j * self.batch_size: (j + 1) * self.batch_size]
                    else:
                        current_batch = shuffle[j * self.batch_size:]

                    a = self.A_train[current_batch]
                    b = self.B_train[current_batch]
                    feed_dict = {self.RealA: a, self.RealB: b, self.isTrain: True}

                    begin = time.time()
                    step = self.step.eval()

                    self.sess.run(self.D_train, feed_dict=feed_dict)

                    self.sess.run(self.G_train, feed_dict=feed_dict)
                    # _, D_summary = self.sess.run([self.D_train, self.D_summary], feed_dict=feed_dict)
                    #
                    # self.writer.add_summary(D_summary, step)
                    #
                    # _, G_summary = self.sess.run([self.G_train, self.G_summary], feed_dict=feed_dict)
                    #
                    # self.writer.add_summary(G_summary, step)

                    # print('Time Per Step: ', format(time.time() - begin, '.3f'), end=' ')

                    if j % logger_frequency == 0:
                        D_loss, G_loss, L_rem, L_vis, L_percept, L_res, L_illum, L_cross, L_adv = self.sess.run(
                            loss_operations, feed_dict=feed_dict)
                        trial_image_idx = np.random.randint(total_image_count)
                        a = self.A_train[trial_image_idx]
                        b = self.B_train[trial_image_idx]

                        if a.ndim == 3:
                            a = np.expand_dims(a, axis=0)

                        if b.ndim == 3:
                            b = np.expand_dims(b, axis=0)

                        feed_dict = {self.RealA: a, self.RealB: b, self.isTrain: False}
                        # img_summary = self.sess.run(self.image_summary, feed_dict=feed_dict)
                        # self.writer.add_summary(img_summary, step)

                        # GT_residual, Fake_residual, GT_illum, Fake_illum = self.sess.run(
                        #     [self.GT_residual, self.Fake_residual, self.GT_illum, self.Fake_illum], feed_dict=feed_dict)
                        # print("j: ", j)
                        # print("shape of D_loss: ", D_loss.shape, " value of D_loss: ", D_loss)
                        # print("shape of G_loss: ", G_loss.shape, " value of G_loss: ", G_loss)
                        # print("shape of L_rem: ", L_rem.shape, " value of L_rem: ", L_rem)
                        # print("shape of L_vis: ", L_vis.shape, " value of L_vis: ", L_vis)
                        # print("shape of L_percept: ", L_percept.shape, " value of L_percept: ", L_percept)
                        # print("shape of L_res: ", L_res.shape, " value of L_res: ", L_res)
                        # print("shape of L_illum: ", L_illum.shape, " value of L_illum: ", L_illum)
                        # print("GT_illum: ", GT_illum)
                        # print("Fake_illum: ", Fake_illum)
                        # print("shape of tf.abs(self.GT_illum-self.Fake_illum):",
                        #       self.sess.run(tf.abs(self.GT_illum - self.Fake_illum), feed_dict=feed_dict))
                        # print("shape of L_cross: ", L_cross.shape, " value of L_cross: ", L_cross)
                        # print("shape of L_adv: ", L_adv.shape, " value of L_adv: ", L_adv)
                        line = 'Batch: %d,\tD_Loss: %.3f,\tG_Loss: %.3f,\tL_rem: %.3f,\tL_res: %.3f,\tL_illum: %.3f,\tL_cross: %.3f,\tL_adv: %.3f' % (
                            j, D_loss, G_loss, L_rem, L_res, L_illum, L_cross, L_adv)
                        print(line)

                        # print(self.RealA)
                        # print('B')
                        # print(self.Fake_coarse)
                        # print('C')
                        # print(self.Fake_residual)

                    # The variable `step` counts both D and G updates as individual steps.
                    # The variable `G_D_step` counts one D update followed by a G update
                    # as a single step.
                    G_D_step = step // 2
                    # print('GD', G_D_step, 'val', val_threshold, '\n')

                    if (val_threshold > G_D_step) and (j % val_frequency == 0):
                        self.validate()

                if not os.path.exists(self.ckpt_dir):
                    os.makedirs(self.ckpt_dir)
                if i % 1 == 0 & i < self.epochs:
                    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
                    model_name = 'deshadow_gan_{:s}.ckpt'.format(str(train_start_time))
                    model_save_path = os.path.join(self.ckpt_dir, model_name)
                    self.saver.save(sess=self.sess, save_path=model_save_path, global_step=i)

            train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
            model_name = 'deshadow_gan_{:s}.ckpt'.format(str(train_start_time))
            model_save_path = os.path.join(self.ckpt_dir, model_name)
            self.saver.save(sess=self.sess, save_path=model_save_path, global_step=self.epochs)
            print("Training Done.")
            """        
                model_save_dir = 'model'
                if not ops.exists(model_save_dir):
                    os.makedirs(model_save_dir)
                if i%30==0&i<self.epochs:
                    train_start_time = time.strftime(
                        '%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
                    model_name = 'deshadow_gan_{:s}.ckpt'.format(str(train_start_time))
                    model_save_path = ops.join(model_save_dir, model_name)
                    self.saver.save(sess=self.sess, save_path=model_save_path,
                               global_step=i)
            #model_save_dir = 'model/deshadow_gan_tensorflow10'
            
            train_start_time = time.strftime(
            '%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
            model_name = 'deshadow_gan_{:s}.ckpt'.format(str(train_start_time))
            model_save_path = ops.join(model_save_dir, model_name)

            self.saver.save(sess=self.sess, save_path=model_save_path,
                               global_step=self.epochs)
            """

    def validate(self):

        total_ssim = 0
        total_psnr = 0
        psnr_weight = 1 / 20
        ssim_weight = 1
        val_image_count = len(self.A_val)

        for i in range(val_image_count):
            x = np.expand_dims(self.A_val[i], axis=0)
            feed_dict = {self.RealA: x, self.isTrain: False}
            generated_B = self.Fake_final.eval(feed_dict=feed_dict)

            print('Validation Image', i, end='\r')

            generated_B = (((generated_B[0] + 1) / 2) * 255).astype(np.uint8)
            real_B = (((self.B_val[i] + 1) / 2) * 255).astype(np.uint8)

            psnr = compare_psnr(real_B, generated_B)
            ssim = compare_ssim(real_B, generated_B, multichannel=True)

            total_psnr = total_psnr + psnr
            total_ssim = total_ssim + ssim

        average_psnr = total_psnr / val_image_count
        average_ssim = total_ssim / val_image_count

        score = average_psnr * psnr_weight + average_ssim * ssim_weight

        if (score > self.score_best):

            self.score_best = score

            self.saver.save(self.sess, os.path.join(self.ckpt_dir, 'gan'), global_step=self.step.eval())
            line = 'Better Score: %.6f, PSNR: %.6f, SSIM: %.6f' % (score, average_psnr, average_ssim)
            print(line)

            with open(os.path.join(self.ckpt_dir, 'logs.txt'), 'a') as f:
                line += '\n'
                f.write(line)

            if self.save_samples:

                try:
                    image_list = os.listdir(self.sample_image_dir)
                except:
                    print('Sample images not found. Terminating program')
                    exit(0)

                for i, file in enumerate(image_list, 1):
                    print('Sample Image', i, end='\r')

                    x = cv2.imread(os.path.join(self.sample_image_dir, file), 1)
                    x = (x / 255) * 2 - 1
                    x = np.reshape(x, (1, 256, 256, 3))

                    feed_dict = {self.RealA: x, self.isTrain: False}
                    img = self.Fake_coarse.eval(feed_dict=feed_dict)

                    img = img[0, :, :, :]
                    img = (((img + 1) / 2) * 255).astype(np.uint8)
                    cv2.imwrite(os.path.join(self.ckpt_dir, file), img)

    def inference(self, input_dir, result_dir):

        input_list = os.listdir(input_dir)

        if not os.path.exists(result_dir):
            os.mkdir(result_dir)

        print('Loading Model')
        self.build_model()
        print('Model Loaded')

        with tf.compat.v1.Session() as self.sess:

            init_op = tf.compat.v1.global_variables_initializer()
            self.sess.run(init_op)

            print('Loading Checkpoint')
            ckpt = tf.train.latest_checkpoint(self.ckpt_dir)
            print(self.ckpt_dir)
            # self.saver.restore(self.sess, save_path=self.ckpt_dir)
            self.saver.restore(self.sess, ckpt)
            self.step = tf.compat.v1.train.get_or_create_global_step()
            print('Checkpoint Loaded')

            for i, img_file in enumerate(input_list, 1):
                img = cv2.imread(os.path.join(input_dir, img_file), 1)
                img = cv2.resize(img, (256, 256))

                print('Processing image', i, end='\r')

                img = ((np.expand_dims(img, axis=0) / 255) * 2) - 1
                feed_dict = {self.RealA: img, self.isTrain: False}
                generated_B = self.Fake_coarse.eval(feed_dict=feed_dict)
                generated_B = (((generated_B[0] + 1) / 2) * 255).astype(np.uint8)

                cv2.imwrite(os.path.join(result_dir, img_file), generated_B)

            print('Done.')
