import tensorflow as tf
import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt
from matplotlib import colors
import keras.backend as K

sess = tf.InteractiveSession()

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# pixel werte auf 0 bis 1 skalieren
train_images = train_images / 255.0
test_images = test_images / 255.0


# one hot encoding of labels
def one_hot_encode(a, length):
    temp = np.zeros((a.shape[0], length))
    temp[np.arange(a.shape[0]), a] = 1.0
    return temp

# one hot one cold encoding of labels
def one_hot_one_cold_encode(a, length):
    temp = np.ones((a.shape[0], length))
    temp = temp * -1
    temp[np.arange(a.shape[0]), a] = 1.0
    return temp


labels_size = 10

#encoding für labels anwenden
train_numeric_labels = train_labels
test_numeric_labels = test_labels

train_labels_one_hot = one_hot_encode(train_labels, labels_size)
test_labels_one_hot = one_hot_encode(test_labels, labels_size)

train_labels_one_hot_one_cold = one_hot_one_cold_encode(train_labels, labels_size)
test_labels_one_hot_one_cold = one_hot_one_cold_encode(test_labels, labels_size)

#dataset infos
(ds_size,image_size,_) = train_images.shape
ds_test_size = int(test_labels.shape[-1])

#augmentation
rauschen = True #False ; True
#loading of Modell
model_to_load = "model_step_0_acc_0.9187999963760376" # "None"; "model_step_0_acc_0.9187999963760376"
#learning infos
learning_rate = 0.1
steps_number = 1551
batch_size = 200
#pruning
pruning_loss = True #False ; True
#info
display_model = True #False ; True
display_pruning = True #False ; True

train_images = np.reshape(train_images, [-1, image_size*image_size])
test_images = np.reshape(test_images, [-1, image_size*image_size])


if rauschen :
        # add rauschen, else minist is too easy
        # gaus um 0 std 0.2
        rausch = np.random.normal(0.1,0.2,(image_size*image_size))

        train_images = train_images + rausch
        test_images = test_images + rausch

        #salt/peper 0.2
        rausch_count = int(image_size*image_size*0.15)
        rausch_index = np.random.randint(0,image_size*image_size,rausch_count)
        rausch = np.zeros((image_size*image_size))
        #salt
        rausch[rausch_index[:rausch_count//2]] = 1
        #peper
        rausch[rausch_index[rausch_count//2:]] = -1

        train_images = train_images + rausch
        test_images = test_images + rausch

        # input images with between 0 and 1
        train_images = np.clip(train_images, 0,1)
        test_images = np.clip(test_images, 0,1)

# pixel werte auf -1 bis 1 skalieren
train_images = train_images * 2 - 1
test_images = test_images * 2 - 1

# create dataset objects from the arrays
dx = tf.data.Dataset.from_tensor_slices(train_images)
dy = tf.data.Dataset.from_tensor_slices(train_labels_one_hot_one_cold)
#dy = tf.data.Dataset.from_tensor_slices(train_labels_one_hot)


batches = tf.data.Dataset.zip((dx, dy)).shuffle(30000).batch(batch_size)

test_labels = test_labels_one_hot_one_cold
#test_labels = test_labels_one_hot


# create a one-shot iterator
iterator = batches.make_initializable_iterator()
# extract an element
next_element = iterator.get_next()


def main():
        model = create_base_model(image_size*image_size,10)
        model_train = create_train_model(model)
        model_eval = create_evaluation_model(model)
        model_prun = create_pruning_model(model)

        saver = tf.train.Saver()
        load_or_init_model(saver)

        train_model(model_train,model_eval,learning_rate, steps_number)

        accuracy = evaluate_model(model_eval)

        if display_model:
                display_model_with_samples(model_prun, 1)

        important_weights = calculate_important_weights(model_prun,1000)
        
        if display_pruning:
                display_important_weights(important_weights)

        pruning_step = 0
        while True:
                save_model(model,f"step_{pruning_step}_acc_{accuracy}",saver)
                pruning_step += 1
                prune_model(model_prun,important_weights,0.6)
                train_model(model_train,model_eval,learning_rate, steps_number//3)
                accuracy = evaluate_model(model_eval)
                important_weights = calculate_important_weights(model_prun,1000)
                if display_pruning:
                        display_important_weights(important_weights)

def create_base_model(inputs, outputs):
        x = tf.placeholder(tf.float32, shape=(None, inputs), name="input")
        tf.add_to_collection("layer_out",x)
        y, mask = connection(x)

        y, y_no_act, weights, mask = fc_layer(y, 36, activation=tf.nn.tanh)

        y, mask = connection(y)

        y, y_no_act, weights, mask = fc_layer(y, 25, activation=tf.nn.tanh)

        y, mask = connection(y)

        y, y_no_act, weights, mask = fc_layer(y, outputs, activation=tf.nn.tanh)

        y, mask = connection(y)

        return (x ,y)

def create_train_model(model):
        x,y = model
        
        with tf.variable_scope("train"):
                ground_truth = tf.placeholder(tf.float32, (None, y.shape[-1]),name="ground_truth")
                tf.add_to_collection("train_labels", ground_truth)

                with tf.variable_scope("loss"):
                        loss = tf.losses.mean_squared_error(ground_truth, y)
                        tf.add_to_collection("train_losses", loss)
                # Training step
                learning_rate = tf.placeholder(tf.float32, None,name="learning_rate")
                train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        return x, ground_truth, loss, train_op, learning_rate

def create_pruning_model(model):
        x,y = model
        
        with tf.variable_scope("prun"):
                ground_truth = tf.placeholder(tf.float32, (None, y.shape[-1]),name="ground_truth")
                tf.add_to_collection("prun_labels", ground_truth)

                with tf.variable_scope("loss"):
                        if pruning_loss:
                                minimum = tf.reduce_min(y)
                                out = tf.subtract(y,minimum)
                                masked_out = tf.multiply(ground_truth,out)
                                loss = tf.reduce_max(masked_out)
                        else:
                                loss = tf.losses.mean_squared_error(ground_truth, y)
                        
                        tf.add_to_collection("prun_losses", loss)

                with tf.variable_scope("gradients"):
                        layer_weights = tf.get_collection("layer_weights")
                        connection_out = tf.get_collection("connection_out")
                        for weights in layer_weights:
                                weight_grad = tf.gradients(loss, weights)
                                tf.add_to_collection("weight_grads", weight_grad)
                        for layer_in in connection_out:
                                input_grad = tf.gradients(loss, layer_in)
                                tf.add_to_collection("input_grads", input_grad)
                        
        return x, ground_truth, loss

def create_evaluation_model(model):
        x,y = model
        
        with tf.variable_scope("eval"):
                ground_truth = tf.placeholder(tf.float32, (None, y.shape[-1]),name="ground_truth")
                tf.add_to_collection("eval_labels", ground_truth)

                with tf.variable_scope("accuracy"):
                        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(ground_truth, 1))
                        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                        tf.add_to_collection("evaluations", accuracy)
                        
        return x, ground_truth, [accuracy]

def load_or_init_model(saver):
        #saver.restore(sess, "./models/model.ckpt")
        try:
                saver.restore(sess, f"./models/{model_to_load}.ckpt")
                print("======-- model initiliced --======")
        except:
                print("========-- Warning! --========")
                print("=    failed to load model    =")
                print("=     initiliced random      =")
                print("========-- Warning! --========")
                sess.run(tf.global_variables_initializer())

def train_model(model_train, model_eval, learning_rate, steps_number):
        x_train, gt_train, loss, training_op, lr = model_train
        x_eval, gt_eval, [accuracy] = model_eval

        # Run the training
        sess.run(iterator.initializer)

        for step in range(steps_number):
                # get batch of images and labels
                (batch_x,batch_y) = sess.run(next_element)
                feed_dict_train = {x_train: batch_x, gt_train: batch_y, lr: learning_rate}

                # Run the training step
                training_op.run(feed_dict=feed_dict_train)
                # Print the accuracy progress on the batch every 100 steps
                if step%100 == 0:
                        feed_dict_eval = {x_eval: batch_x, gt_eval: batch_y}
                        train_accuracy = accuracy.eval(feed_dict=feed_dict_eval)
                        print("Step %d, training batch accuracy %g %%"%(step, train_accuracy*100))

                if (step + 1) % (ds_size // batch_size) == 0 and step > 0:
                        sess.run(iterator.initializer)

def evaluate_model(model_eval):
        x_eval, gt_eval, [accuracy] = model_eval

        feed_dict_eval = {x_eval: test_images, gt_eval: test_labels}
        test_accuracy = accuracy.eval(feed_dict=feed_dict_eval)
        print("Test accuracy: %g %%"%(test_accuracy*100))
        return test_accuracy

def save_model(model,name,saver):
        save_path = saver.save(sess, f"./models/model_{name}.ckpt")
        print("Model saved in path: %s" % save_path)


def display_model_with_samples(model_prun, sampels):
        weight_grads = tf.get_collection("weight_grads")
        input_grads = tf.get_collection("input_grads")
        connection_out = tf.get_collection("connection_out")

        #Display some sample images
        for i in range(sampels):
                print(i, " from ", sampels)
                #choose random sample images
                test_image_nr = np.random.randint(0,ds_test_size)
                test_image_nr = 11 # select fixed image for testing

                image = np.reshape(test_images[test_image_nr],[1,-1])
                if pruning_loss:
                        label_mask = np.reshape(test_labels_one_hot[test_image_nr],[1,-1])
                else:
                        label_mask = np.reshape(test_labels_one_hot_one_cold[test_image_nr],[1,-1])

                x, ground_truth, loss = model_prun

                feed_dict = {x: image, ground_truth: label_mask}
                
                # calculate the output (feature map) of each layer
                # -> perform a layer vice forward pass
                outs = []
                for outputs in connection_out:
                        out = sess.run(outputs, feed_dict=feed_dict)
                        outs.append(out)

                        out = out[0]
                       
                       #display the output
                        print(outputs.name)
                        print(outputs.shape)
                        plt.title(outputs.name)

                        x = np.sqrt(len(out))
                        y = np.sqrt(len(out))
                        plt.xticks(np.arange(0, x))
                        plt.yticks(np.arange(0, y))

                        if x * y == len(out):
                                out = np.reshape(out,[int(x),int(y)])
                                #print("Graident of inputs\n=", grad)
                                plt.imshow(out, cmap='binary')
                                plt.colorbar()
                                plt.show()
                        elif 2 * 5 == len(out):
                                out = np.reshape(out,[2,5])
                                #print("Graident of inputs\n=", grad)
                                plt.imshow(out, cmap='binary')
                                plt.colorbar()
                                plt.show()
                        else:
                                print("Error:",x,"*",y,"=",x * y," != ",len(grad))
                
                #calculate impact of weights directly on final loss
                for weight_grad in weight_grads:
                        weight_grad = weight_grad[0]
                        grad = sess.run(weight_grad, feed_dict=feed_dict)
                        
                        print(weight_grad.name)
                        print(weight_grad.shape)

                        x = np.sqrt(len(grad))
                        y = np.sqrt(len(grad))

                        if x * y == len(grad):
                                show_images(grad,[int(x),int(y)],weight_grad.name)
                        else:
                                print("Error:",x,"*",y,"=",x * y," != ",len(grad))

                #calculate impact of input directly on final loss
                for input_grad in input_grads:
                        input_grad = input_grad[0]
                        grad = sess.run(input_grad, feed_dict=feed_dict)
                        grad = grad[0]

                        print(input_grad.name)
                        print(input_grad.shape)
                        plt.title(input_grad.name)

                        x = np.sqrt(len(grad))
                        y = np.sqrt(len(grad))
                        plt.xticks(np.arange(0, x))
                        plt.yticks(np.arange(0, y))

                        if x * y == len(grad):
                                grad = np.reshape(grad,[int(x),int(y)])
                                plt.imshow(grad, cmap='binary')
                                plt.colorbar()
                                plt.show()
                        elif 2 * 5 == len(grad):
                                grad = np.reshape(grad,[2,5])
                                plt.imshow(grad, cmap='binary')
                                plt.colorbar()
                                plt.show()
                        else:
                                print("Error:",x,"*",y,"=",x * y," != ",len(out))
                        
                #calculate abs impact of input directly on final loss
                for input_grad in input_grads:
                        input_grad = input_grad[0]
                        grad = sess.run(input_grad, feed_dict=feed_dict)
                        grad = np.abs(grad[0])

                        print(input_grad.name)
                        print(input_grad.shape)
                        plt.title(input_grad.name)

                        x = np.sqrt(len(grad))
                        y = np.sqrt(len(grad))
                        plt.xticks(np.arange(0, x))
                        plt.yticks(np.arange(0, y))

                        if x * y == len(grad):
                                grad = np.reshape(grad,[int(x),int(y)])
                                plt.imshow(grad, cmap='binary')
                                plt.colorbar()
                                plt.show()
                        elif 2 * 5 == len(grad):
                                grad = np.reshape(grad,[2,5])
                                plt.imshow(grad, cmap='binary')
                                plt.colorbar()
                                plt.show()
                        else:
                                print("Error:",x,"*",y,"=",x * y," != ",len(out))
                        
                #--------mask output-----------
                #calculate from back to front the impact of input on output
                #mask all not important inputs while reverse calculation
                #so only important connection have high impacts
                #TODO wenn using mask gradient diffrent then wen using connections
                input_importance = []
                for input_grad in input_grads:
                        input_grad = input_grad[0]
                        grad = sess.run(input_grad, feed_dict=feed_dict)
                        grad = np.abs(grad[0])


                        minimum = np.min(grad)
                        maximum = np.max(grad)

                        if minimum < maximum:
                                importance = grad - minimum
                                importance = importance / (maximum - minimum)
                        else:
                                importance = grad - minimum
                        input_importance.append(importance)

                        print(input_grad.name)
                        print(input_grad.shape)
                        plt.title(input_grad.name)

                        x = np.sqrt(len(importance))
                        y = np.sqrt(len(importance))
                        plt.xticks(np.arange(0, x))
                        plt.yticks(np.arange(0, y))

                        if x * y == len(importance):
                                importance = np.reshape(importance,[int(x),int(y)])
                                plt.imshow(importance, cmap='binary')
                                plt.colorbar()
                                plt.show()
                        elif 2 * 5 == len(importance):
                                importance = np.reshape(importance,[2,5])
                                plt.imshow(importance, cmap='binary')
                                plt.colorbar()
                                plt.show()
                        else:
                                print("Error:",x,"*",y,"=",x * y," != ",len(grad))

                for importance_1, importance_2 in zip(input_importance[:-1],input_importance[1:]):
                        print(len(importance_1))
                        print(len(importance_2))
                        weight_importance = []
                        for importance in importance_2:
                                singel_weight_importance = importance_1 * importance
                                weight_importance.append(singel_weight_importance)

                        weight_importance = np.asarray(weight_importance).T
                        print(weight_importance.shape)
                        x = np.sqrt(len(weight_importance))
                        y = np.sqrt(len(weight_importance))

                        if x * y == len(weight_importance):
                                show_images(weight_importance,[int(x),int(y)],"weight_importance")
                        else:
                                print("Error:",x,"*",y,"=",x * y," != ",len(weight_importance))
                        

def calculate_important_weights(model_prun,samples):
        input_grads = tf.get_collection("input_grads")
        cummulated_weight_importance=[]
        import time

        for i in range(samples):#ds_test_size):
                if display_pruning:
                        print(i, " from ", samples)

                test_image_nr = np.random.randint(0,ds_test_size)

                image = np.reshape(test_images[test_image_nr],[1,-1])
                if pruning_loss:
                        label_mask = np.reshape(test_labels_one_hot[test_image_nr],[1,-1])
                else:
                        label_mask = np.reshape(test_labels_one_hot_one_cold[test_image_nr],[1,-1])

                x, ground_truth, loss = model_prun

                feed_dict = {x: image, ground_truth: label_mask}

                input_importance = []
                for input_grad in input_grads:
                        input_grad = input_grad[0]
                        grad = sess.run(input_grad, feed_dict=feed_dict)
                        grad = np.abs(grad[0])


                        minimum = np.min(grad)
                        maximum = np.max(grad)

                        #Min Max norm of Gradients
                        if minimum < maximum:
                                importance = grad - minimum
                                importance = importance / (maximum - minimum)
                        else:
                                importance = grad - minimum
                        
                        input_importance.append(importance)


                all_weight_importance=[]
                for importance_1, importance_2 in zip(input_importance[:-1],input_importance[1:]):
                        weight_importance = []
                        for importance in importance_2:
                                singel_weight_importance = importance_1 * importance
                                weight_importance.append(singel_weight_importance)
                        weight_importance = np.asarray(weight_importance).T
                        all_weight_importance.append(weight_importance)
                

                if len(cummulated_weight_importance) == 0:
                        cummulated_weight_importance = np.asarray(all_weight_importance)
                else:
                        cummulated_weight_importance += np.asarray(all_weight_importance)
        
        ## Mask out pruned weights
        layer_masks = tf.get_collection("layer_masks")
        layer_masks_values=[]
        for layer_mask in layer_masks:
                layer_mask_value = layer_mask.eval()
                layer_masks_values.append(layer_mask_value)
        cummulated_weight_importance = cummulated_weight_importance * np.asarray(layer_masks_values)
        return cummulated_weight_importance
                
def display_important_weights(cummulated_weight_importance):
        for weight_importance_sum in cummulated_weight_importance:

                x = np.sqrt(len(weight_importance_sum))
                y = np.sqrt(len(weight_importance_sum))
                if x * y == len(weight_importance_sum):
                        show_images(weight_importance_sum,[int(x),int(y)])
                else:
                        print("Error:",x,"*",y,"=",x * y," != ",len(weight_importance_sum)) 

def prune_model(prune_model,important_weights,sparcification_factor):
        layer_masks = tf.get_collection("layer_masks")
        # Calculate pruning mask
        # Go through every layer
        for important_weight,layer_mask in zip(important_weights,layer_masks):
                layer_mask_value = layer_mask.eval()

                # Go through every neuron
                layer_mask_value = layer_mask_value.T
                important_weight = important_weight.T
                masks = []
                for weight,weight_mask in zip(important_weight,layer_mask_value):

                        maximum = np.max(weight)
                        ##else here maybe epty list
                        if maximum > 0 :
                                sparcity = len(weight_mask) / sum(weight_mask)
                                minimum = np.min(weight[np.nonzero(weight)])
                                weight = weight - minimum

                                median = np.median(weight[np.nonzero(weight)])
                                mask = weight > pow(sparcification_factor,sparcity) * median
                        else:
                                mask = weight > 1
                        masks.append(mask)

                masks = np.asarray(masks).T

                if display_pruning:
                        display_puning_masks(masks)
                
                print(sum(masks.flatten())," from ", len(masks.flatten())," sparsity: ", sum(masks.flatten())/len(masks.flatten()))
                layer_mask.load(masks, sess)


def display_puning_masks(masks):
        x = np.sqrt(len(masks))
        y = np.sqrt(len(masks))
        if x * y == len(masks):
                show_images(masks,[int(x),int(y)])
        else:
                print("Error:",x,"*",y,"=",x * y," != ",len(masks))
                

def connection(x, name = None):
        with tf.variable_scope(name, "connection",[x]):
                print(tf.get_variable_scope().name)
                print(x.shape)
                mask = tf.get_variable("mask",shape=x.shape[-1],initializer=tf.constant_initializer(1),trainable=False)
                tf.add_to_collection("connection_masks", mask)
                y = tf.multiply(x, mask)
                tf.add_to_collection("connection_out", y)
                print(y.shape)

        return y, mask

def fc_layer(x, outputs, activation = tf.nn.sigmoid, name = None):
        with tf.variable_scope(name, "fc_layer", [x]):
                print(tf.get_variable_scope().name)
                print(x.shape)
                weights = tf.get_variable("weights", [x.shape[-1], outputs])
                tf.add_to_collection("layer_weights", weights)
                print(weights.shape)

                biases = tf.get_variable("biases", [outputs])
                tf.add_to_collection("layer_biases", biases)
                print(biases.shape)

                mask = tf.get_variable("mask",[x.shape[-1], outputs],initializer=tf.constant_initializer(1),trainable=False)
                tf.add_to_collection("layer_masks", mask)
                print(mask.shape)
                
                masked_weights = tf.multiply(weights, mask)
                y_no_activation = tf.nn.bias_add(tf.matmul(x, masked_weights), biases)
                tf.add_to_collection("layer_out_no_activation", y_no_activation)
                
                if activation == None:
                        tf.add_to_collection("layer_out", y_no_activation)
                        print(y_no_activation.shape)
                        return y_no_activation, y_no_activation, weights, mask
                else:
                        y = activation(y_no_activation)
                        tf.add_to_collection("layer_out", y)
                        print(y.shape)
                        return y, y_no_activation, weights, mask


def show_images(grad,image_shape,titel = 'Multiple images'):
        size, neurons = grad.shape
        Nc = 5
        Nr = int(neurons/Nc)
        cmap = 'binary'#'coolwarm_r'#'hot'#'jet'#"cool"

        fig, axs = plt.subplots(Nr, Nc)
        fig.suptitle(titel)

        images = []
        for i in range(Nr):
            for j in range(Nc):
                # Generate data with a range that varies from one plot to the next.
                neuron_grads = grad[:,i*j]
                data = np.reshape(neuron_grads,image_shape)
                images.append(axs[i, j].imshow(data, cmap=cmap))
                axs[i, j].set_xticks(np.arange(0, image_shape[0]))
                axs[i, j].set_yticks(np.arange(0, image_shape[1]))
                axs[i, j].label_outer()

        # Find the min and max of all colors for use in setting the color scale.
        vmin = min(image.get_array().min() for image in images)
        vmax = max(image.get_array().max() for image in images)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        for im in images:
            im.set_norm(norm)

        fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=.1)


        # Make images respond to changes in the norm of other images (e.g. via the
        # "edit axis, curves and images parameters" GUI on Qt), but be careful not to
        # recurse infinitely!
        def update(changed_image):
            for im in images:
                if (changed_image.get_cmap() != im.get_cmap()
                    or changed_image.get_clim() != im.get_clim()):
                    im.set_cmap(changed_image.get_cmap())
                    im.set_clim(changed_image.get_clim())


        for im in images:
            im.callbacksSM.connect('changed', update)

        plt.show()

if __name__ == "__main__":
    main()