from random import randint
from numpy import array
from numpy import argmax
from numpy import array_equal
import tensorflow as tf
import numpy as np
import random
import math
#from matplotlib import pyplot as plt
import os
import copy
import time

from influxdb import InfluxDBClient
from tensorflow.python.layers import core as layers_core

from tensorflow.python import debug as tf_debug

import pandas as pd




def generate_sin_wave(no_of_samples):
    """pass the number of samples you want to generate"""
    samples=no_of_samples/10

    time          = np.arange(0, samples, 0.1)
    sin_wave_data = np.sin(time)

    return sin_wave_data
def read_input(arguments):

    # timestamps
    end=start_time
    start=end-arguments.limit*arguments.interval

    ct = 0
    timestamps = []
    values = []
    lasttimestamps = []
    client = InfluxDBClient(host='localhost',port='8086',database=arguments.data_base)

    if arguments.multiple_tables:
        for tab in arguments.tables:
            query = "SELECT time,"+arguments.parameter+" FROM \""+tab+"\" WHERE time > "+str(int(start))+"s AND time < "+str(int(end))+"s;"
            #query = "SELECT time,"+arguments.parameter+" FROM \""+tab+"\" ORDER BY DESC LIMIT %d"%(arguments.limit)
            output = client.query(query,epoch='s')
            output = np.array(output.raw["series"][0]["values"]).transpose()
            tm  = list(output[0])
            val = list(output[1])
            print("value",len(val))
            lasttimestamps.append(tm[-1])
            timestamps.append(tm)
            ct=ct+1
                #fills missing timestamps with mean value
            lasttm = tm[-1]
            i = 0
            while tm[i] != lasttm:
                if i != 0 and tm[i] - tm[i-1] >= 2*arguments.interval:
                    mean = (val[i] + val[i-1])/2
                    missingPoints = int((tm[i] - tm[i-1])/arguments.interval)
                    for j in range(missingPoints-1):
                        val.insert((i+1), mean)
                        i = i+1
                i = i+1
            values.append(val)
        while True:
            for i in range(len(values)):
                for j in range(len(values)):
                    if len(values[i]) > len(values[j]):
                        values[i].pop(0)
            if all(len(v) == len(values[0]) for v in values):
                break

        datapoints = np.vstack((values))
        print (datapoints.shape)
        datapoints = datapoints.T
        datapoints = pd.DataFrame(datapoints)
        return datapoints, lasttimestamps
    else:
        for param in arguments.parameter:
            query = "SELECT time,"+param+" FROM \""+arguments.tables[0]+"\" WHERE time > "+str(int(start))+"s AND time < "+str(int(end))+"s;"
            #query = "SELECT time,"+param+" FROM \""+arguments.tables+"\" ORDER BY DESC LIMIT %d"%(arguments.limit)
            output = client.query(query,epoch='s')

            output = np.array(output.raw["series"][0]["values"]).transpose()
            tm  = list(output[0])
            val = list(output[1])
            print("value",len(val))
            lasttimestamps.append(tm[-1])
            timestamps.append(tm)
            ct=ct+1
                #fills missing timestamps with mean value
            lasttm = tm[-1]
            i = 0
            while tm[i] != lasttm:
                if i != 0 and tm[i] - tm[i-1] >= 2*arguments.interval:
                    mean = (val[i] + val[i-1])/2
                    missingPoints = int((tm[i] - tm[i-1])/arguments.interval)
                    for j in range(missingPoints-1):
                        val.insert((i+1), mean)
                        i = i+1
                i = i+1
            values.append(val)
        while True:
            for i in range(len(values)):
                for j in range(len(values)):
                    if len(values[i]) > len(values[j]):
                        values[i].pop(0)
            if all(len(v) == len(values[0]) for v in values):
                break
        datapoints = np.vstack((values))
        print (datapoints.shape)
        datapoints = datapoints.T
        datapoints = pd.DataFrame(datapoints ,columns =arguments.parameter)
        return datapoints, lasttimestamps
#write function

def write_data(arguments,predicted_list):
    import os
    ti=arguments.interval
    np_predicted_list = np.array(predicted_list)
    print( "writing")
    #arguments.multiple_tables =True
    if arguments.multiple_tables:
        for i in range(0, len(predicted_list[0])):
            var = np_predicted_list[:,i]
            t=arguments.last_ts[i]
            for j in var:
                t = int(t)+ti
                print( t)
                os.system("docker-compose exec influxdb curl -i -silent -XPOST 'http://localhost:8086/write?db=%s&precision=s' --data-binary '%s,host=%s value=%s %d'" % (arguments.data_base,arguments.w_measurement[i],arguments.host,j,t))
                print("docker-compose exec influxdb curl -i -silent -XPOST 'http://localhost:8086/write?db=%s&precision=s' --data-binary '%s,host=%s value=%s %d'" % (arguments.data_base,arguments.w_measurement[i],arguments.host,j,t))

        return
    else:
        t=arguments.last_ts[0]
        for i in range(0, len(predicted_list[0])):
            var = np_predicted_list[:,i]

            for j in var:
                t = int(t)+ti
                print (t)
                os.system("docker-compose exec influxdb curl -i -silent -XPOST 'http://localhost:8086/write?db=%s&precision=s' --data-binary '%s,host=%s %s=%s %d'" % (arguments.data_base,arguments.w_measurement[0],arguments.host,arguments.w_parameters[i],j,t))
        return



def seperate_train_test_datasets(x, y, batch_size, input_seq_len, output_seq_len, number_of_test_batch_sets):

    last_test_datapoint_index = input_seq_len+output_seq_len+(batch_size*number_of_test_batch_sets)
    total_datapoints = len(x)

    #checking the ratio between train and test dataset size
    if total_datapoints*.25 <= last_test_datapoint_index:
        import warnings
        warnings.warn("Number of test datapoints is more than 25% of total number of datapoints")
    assert (total_datapoints*.5 >= last_test_datapoint_index), "Number of test datapoints is more than 50% of total number of datapoints"

    x_test = x[:last_test_datapoint_index]
    y_test = y[:last_test_datapoint_index]

    x_train = x[last_test_datapoint_index:]
    y_train = y[last_test_datapoint_index:]
    return x_train, y_train, x_test, y_test

def generate_train_batches(x, y, batch_size, input_seq_len, output_seq_len,time_major,seq_batch,last_batch_no,n_in_features,n_out_feature):

    import numpy as np
    total_start_points = len(x) - input_seq_len - output_seq_len

    #For creating the batches from sequential or random indices
    if seq_batch:
        if last_batch_no >= total_start_points-batch_size:
            last_batch_no=0
        #Selecting successive indices for creating batches
        start_x_idx = np.arange(last_batch_no, last_batch_no + batch_size)
        last_batch_no +=len(start_x_idx)
    else:
        #Selecting random indices for creating batches
        start_x_idx = np.random.choice(range(total_start_points), batch_size, replace = False)
        last_batch_no = 0

    input_batch_idxs = [(range(i, i+input_seq_len)) for i in start_x_idx]
    input_seq = np.take(x, input_batch_idxs, axis = 0)
    output_batch_idxs = [(range(i+input_seq_len, i+input_seq_len+output_seq_len)) for i in start_x_idx]
    output_seq = np.take(y, output_batch_idxs, axis = 0)

    #Outputs the batches in time major (shape = (time,batchsize,features), if selected)
    if time_major:
        input_seq =(input_seq.reshape(input_seq.shape[0],input_seq.shape[1],n_in_features)).transpose((1,0,2))
        output_seq=(output_seq.reshape(output_seq.shape[0],output_seq.shape[1],n_out_feature)).transpose((1,0,2))
        return input_seq, output_seq, last_batch_no  # in shape: (time_steps, batch_size, feature_dim)
    else:
        input_seq =(input_seq.reshape(input_seq.shape[0],input_seq.shape[1],n_in_features))
        output_seq=(output_seq.reshape(output_seq.shape[0],output_seq.shape[1],n_out_feature))
        return input_seq, output_seq,last_batch_no

def generate_test_batches(x, y, batch_size, input_seq_len, output_seq_len,time_major,n_in_features,n_out_feature):

    import numpy as np
    total_start_points = len(x) - input_seq_len - output_seq_len

    #Selecting random indices for creating batches
    start_x_idx = np.random.choice(range(total_start_points), batch_size)

    input_batch_idxs = [(range(i, i+input_seq_len)) for i in start_x_idx]
    input_seq = np.take(x, input_batch_idxs, axis = 0)
    output_batch_idxs = [(range(i+input_seq_len, i+input_seq_len+output_seq_len)) for i in start_x_idx]
    output_seq = np.take(y, output_batch_idxs, axis = 0)

    #Outputs the batches in time major (shape = (time,batchsize,features), if selected)
    if time_major:
        input_seq =(input_seq.reshape(input_seq.shape[0],input_seq.shape[1],n_in_features)).transpose((1,0,2))
        output_seq=(output_seq.reshape(output_seq.shape[0],output_seq.shape[1],n_out_feature)).transpose((1,0,2))
        return input_seq, output_seq  # in shape: (time_steps, batch_size, feature_dim)
    else:
        input_seq =(input_seq.reshape(input_seq.shape[0],input_seq.shape[1],n_in_features))
        output_seq=(output_seq.reshape(output_seq.shape[0],output_seq.shape[1],n_out_feature))

        return input_seq, output_seq # in shape: (batch_size, time_steps, feature_dim)

def generate_inference_batches (x, y, batch_size, input_seq_len ,time_major,n_in_features,n_out_feature):

    import numpy as np
    output_seq_len =input_seq_len
    total_start_points = len(x) - input_seq_len -output_seq_len

    start_x_idx =np.arange(total_start_points - batch_size ,total_start_points)


    """ for production  we need to pass the last data as input so we can get the next output"""
    if n_in_features != n_out_feature:
        input_batch_idxs_b = [(range(i+input_seq_len, i+input_seq_len+output_seq_len)) for i in start_x_idx]
        last_batch = np.take(x, input_batch_idxs_b, axis = 0)
        print("last batch")
    else:
        last_batch=None

    #Selecting random indices for creating batches

    input_batch_idxs = [(range(i, i+input_seq_len)) for i in start_x_idx]
    input_seq = np.take(x, input_batch_idxs, axis = 0)

    output_batch_idxs = [(range(i+input_seq_len, i+input_seq_len+output_seq_len)) for i in start_x_idx]
    output_seq = np.take(y, output_batch_idxs, axis = 0)

    #Outputs the batches in time major (shape = (time,batchsize,features), if selected)
    if time_major:
        if n_in_features != n_out_feature:
            last_batch =(last_batch.reshape(last_batch.shape[0],last_batch.shape[1],n_in_features)).transpose((1,0,2))
            #last_batch=last_batch[:,-1:,:]
            print(last_batch.shape)

        input_seq =(input_seq.reshape(input_seq.shape[0],input_seq.shape[1],n_in_features)).transpose((1,0,2))
        output_seq=(output_seq.reshape(output_seq.shape[0],output_seq.shape[1],n_out_feature)).transpose((1,0,2))

        return input_seq,output_seq,last_batch  # in shape: (time_steps, batch_size, feature_dim)
    else:
        if n_in_features != n_out_feature:
            last_batch =(last_batch.reshape(last_batch.shape[0],last_batch.shape[1],n_in_features)).transpose((1,0,2))
            #last_batch=last_batch[-1:,:,:]
        input_seq =(input_seq.reshape(input_seq.shape[0],input_seq.shape[1],n_in_features))
        output_seq=(output_seq.reshape(output_seq.shape[0],output_seq.shape[1],n_out_feature))

        return input_seq,output_seq,last_batch # in shape: (batch_size, time_steps, feature_dim)

def batch_data_ploting(previous_data,prediction_data,target_data):

    test_in=previous_data
    pred_outputs=prediction_data
    test_out=target_data

    #previous data encoder input data
    previous_a=test_in[:,0,:]
    previous_b =test_in[-1,:,:]

    previous_in_seq=np.concatenate((previous_a[:-1],previous_b) ,axis=0)

    #Prediction data
    prediction_a=pred_outputs[:,0,:]
    prediction_b =pred_outputs[-1,:,:]

    prediction_seq=np.concatenate((prediction_a[:-1],prediction_b) ,axis=0)

    #real target data
    target_a=test_out[:,0,:]
    target_b =test_out[-1,:,:]

    target_seq=np.concatenate((target_a[:-1],target_b) ,axis=0)

    target_seq=scaler.inverse_transform(target_seq)
    prediction_seq=scaler.inverse_transform(prediction_seq)
    previous_in_seq=scaler.inverse_transform(previous_in_seq)

    l1, = plt.plot(range(len(previous_in_seq),len(previous_in_seq)+len(prediction_seq)), prediction_seq, 'b', label = 'prediction_X')
    l2, = plt.plot(range(len(previous_in_seq),len(previous_in_seq)+len(target_seq)), target_seq, 'y', label = 'actual_data_Y')

    l3, = plt.plot(range(len(previous_in_seq)), previous_in_seq, 'r', label = 'previous_data')
    plt.legend(handles = [l1, l2,l3], loc = 'upper left')
    plt.show()

def last_batch_data_ploting(previous_data,prediction_data,scaler,target_data,time_major,color):

    if time_major:

        actual_previous=previous_data[:,-1,:]
        final_batch_output=prediction_data[:,-1,:]
    else:
        actual_previous=previous_data[-1,:,:]
        final_batch_output=prediction_data[-1,:,:]

    if scaler:

        actual_previous = scaler.inverse_transform(actual_previous)
        final_batch_output = scaler.inverse_transform(final_batch_output)

    if target_data is not None:
        if time_major:

            actual_target=target_data[:,-1,:]
        else :
            actual_target=target_data[-1,:,:]

        if scaler:
             actual_target = scaler.inverse_transform(actual_target)

        l1, = plt.plot(range(len(actual_previous),len(actual_previous)+len(final_batch_output)), final_batch_output, color, label = 'prediction_X')
        l2, = plt.plot(range(len(actual_previous),len(actual_previous)+len(actual_target)), actual_target, 'y', label = 'actual_data_Y')

        l3, = plt.plot(range(len(actual_previous)), actual_previous, 'r', label = 'previous_data')
        plt.legend(handles = [l1, l2,l3], loc = 'upper left')
        plt.show()
    else:
        l1, = plt.plot(range(len(actual_previous),len(actual_previous)+len(final_batch_output)), final_batch_output, 'k', label = 'prediction_X')
        l3, = plt.plot(range(len(actual_previous)), actual_previous, 'r', label = 'previous_data')
        plt.legend(handles = [l1,l3], loc = 'upper left')
        plt.show()



def load_data(arguments):
    #load dataset
    #df = pd.read_excel('./sin_wave.xlsx')
    #df = pd.read_excel('./Incident_monthly_full_data.xlsx')
    df,_ = read_input(arguments)
    #df =pd.read_csv("all_load_average_data.csv" , index_col='time')
    #df=df['netdata.system.load.load1.csv']
    df.fillna(0 ,inplace=True)
    print(df.tail())


    #df.drop(['netdata.system.io.in.csv','netdata.system.io.out.csv','netdata.system.net.received.csv','netdata.system.net.sent.csv'] , inplace =True ,axis =1)
    #df_Y=df[2].values
    #print("target " ,df_Y[:5])
    df_train=df[:].values

    print("data size",len(df))
    X = df_train.reshape(-1,1)
    Y = df_train.reshape(-1,1)

    print("X shape :{} , Y Shape {}".format(X.shape,Y.shape))


    from sklearn.preprocessing import MinMaxScaler
    scaler_y= MinMaxScaler()
    #scaler=None
    #scaler = MinMaxScaler()
    Y =scaler_y.fit_transform(Y)
    X =scaler_y.fit_transform(X)

    x_train=X
    y_train=Y
    x_test =0
    y_test =0


    #Create train and test datasets
    #x_train, y_train, x_test, y_test = seperate_train_test_datasets(X, Y, arguments.batch_size, arguments.input_seq_len, arguments.output_seq_len, arguments.number_of_test_batch_sets)

    return x_train, y_train, x_test, y_test, scaler_y


class parameters(object):
    def __init__(self):
        self.data_base="aadata"
        self.parameter= ["value"] #"cpu_utilization","Disk_utilization","load_average",
                         #"process_running","process_blocked","Free_Ram",
                         #"context_switches","forks_started","proccess_active",
                         #"cpu_iowait"]#,"jitter_average","interrupts" ]
        self.tables = [
            "sqldata"
        ]   #multiple input measurements can be added
        self.w_measurement = [
            "sqldata_Pred2"
        ]   #multiple output measurements can be added
        self.host ="dedusgfa003"
        self.interval = 86400 #seconds
        self.limit = 700 #30000
        self.multiple_tables = False
        self.w_parameters = ["sql_e_Pred"]

        self.decoder_input_as_target=False
        self.use_attention=True
        self.bidir =True
        self.inference = False
        self.time_major = True
        self.seq_batch = False
        self.last_batch_no = 0

        self.n_in_features=1
        self.n_out_feature=1


        self.input_seq_len =10
        self.output_seq_len =10
        self.batch_size =64
        self.number_of_test_batch_sets =0
        self.last_ts = [start_time]
        self.learning_rate = 0.003
        self.lambda_l2_reg = 0.002

        ## Network Parameters
        self.in_n_features=1
        self.out_feature=1
        # size of LSTM Cell
        self.hidden_dim = 128
        # num of input signals
        self.input_dim = self.n_in_features
        # num of output signals
        self.output_dim = self.n_out_feature
        # num of stacked lstm layers
        # gradient clipping - to avoid gradient exploding
        self.GRADIENT_CLIPPING = 2.5
        self.max_gradient_norm = 5.0

        self.number_of_layers=4
        self.keep_prob =1
        self.epochs =1000#100000 #500#10000

        self.keep_prob = 1

        self.ckpt_dir = "checkpoints/"

        self.enc_length = self.input_seq_len
        self.dec_length = self.output_seq_len





class Model(object):
    def __init__(self, hparam):
        self.enc_inp = tf.placeholder(tf.float32, shape=(None ,None,hparam.in_n_features),name="enc_inp")

        #enc_cell=tf.nn.rnn_cell.BasicLSTMCell(hparam.hidden_dim)
        self.enc_seq_len = tf.placeholder(tf.int32, [hparam.batch_size])

        def get_a_cell(hidden_dim, keep_prob):
                    lstm = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim)
                    drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
                    return drop

        with tf.name_scope('lstm'):
            stacked_enc = tf.nn.rnn_cell.MultiRNNCell([get_a_cell(hparam.hidden_dim,hparam.keep_prob) for _ in range(hparam.number_of_layers)])

        #bidirectional RNN
        if hparam.bidir :

            encoder_outputs,all_final_states = tf.nn.bidirectional_dynamic_rnn(
                stacked_enc,stacked_enc,self.enc_inp, time_major=True, dtype=tf.float32)

            encoder_outputs=tf.concat(encoder_outputs, -1)
            if arguments.number_of_layers == 1:
                encoder_state = all_final_states
            else:

            # alternatively concat forward and backward states
                encoder_state = []
                for layer_id in range(arguments.number_of_layers):
                   con_h= tf.concat((all_final_states[0][layer_id].h, all_final_states[1][layer_id].h), 1)
                   con_c= tf.concat((all_final_states[0][layer_id].c, all_final_states[1][layer_id].c), 1)
                   final_states= tf.contrib.rnn.LSTMStateTuple(c=con_c, h=con_h)
                   encoder_state.append(final_states)


                #encoder_state.append(all_final_states[0][layer_id])  # forward
                #encoder_state.append(all_final_states[1][layer_id])  # backward
            encoder_state = tuple(encoder_state)
            with tf.name_scope('lstm_decoder'):
                stacked_decoder = tf.nn.rnn_cell.MultiRNNCell([get_a_cell(hparam.hidden_dim*2,1) for _ in range(hparam.number_of_layers)])
            print("Using BID RNN")

        else:
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(stacked_enc, self.enc_inp,sequence_length=self.enc_seq_len, dtype=tf.float32, time_major=hparam.time_major)
            with tf.name_scope('lstm_decoder'):
                stacked_decoder = tf.nn.rnn_cell.MultiRNNCell([get_a_cell(hparam.hidden_dim,1) for _ in range(hparam.number_of_layers)])

        ##Decoder
        self.decoder_targets = tf.placeholder(tf.float32, shape=(None,None,hparam.out_feature),name='decoder_targets')
        self.decoder_lengths = tf.placeholder(tf.int32, shape=(hparam.batch_size), name="decoder_length")

        ## decoder input as decoder target
        if hparam.decoder_input_as_target:
            print("decoder_input_as_target")
            decoder_inputs =tf.concat(((tf.zeros_like(self.decoder_targets[:1], dtype=tf.float32, name="GO")),self.decoder_targets[:-1]),axis=0)
        else :
            print("enc_inp_input_as_target")
            decoder_inputs =tf.concat(((tf.zeros_like(self.enc_inp[:1], dtype=tf.float32, name="GO")),self.enc_inp[:-1]),axis=0)

        #Decoder Cell
        """def get_a_cell(hidden_dim):# keep_prob):
                    lstm = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim)
                    #drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
                    return lstm

        with tf.name_scope('lstm_decoder'):
            stacked_decoder = tf.nn.rnn_cell.MultiRNNCell([get_a_cell(hparam.hidden_dim) for _ in range(hparam.number_of_layers)])
            """

        #Output layer
        projection_layer = layers_core.Dense(hparam.out_feature, use_bias=False)

        ##Training Decoder
        with tf.variable_scope("decode"):
            # Helper
            helper = tf.contrib.seq2seq.TrainingHelper(decoder_inputs, self.decoder_lengths,name="helper", time_major=hparam.time_major)

            #Attention
            if hparam.use_attention:
                print("Using Attention")
                if hparam.time_major:
                    attention_states =tf.transpose(encoder_outputs, [1, 0, 2])
                else :
                    attention_states =encoder_outputs
                attention_mechanism =tf.contrib.seq2seq.BahdanauAttention(hparam.hidden_dim, attention_states,memory_sequence_length=self.enc_seq_len) # Create an attention mechanism
                stacked_decoder = tf.contrib.seq2seq.AttentionWrapper(stacked_decoder, attention_mechanism,attention_layer_size=hparam.hidden_dim)
                initial_state = stacked_decoder.zero_state(hparam.batch_size, tf.float32).clone(cell_state=encoder_state)
            else:
                initial_state = encoder_state

            decoder = tf.contrib.seq2seq.BasicDecoder(stacked_decoder, helper, initial_state, output_layer=projection_layer)
            final_outputs, _final_state, _final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder,output_time_major=hparam.time_major)
            self.logits = final_outputs.rnn_output
        """
        ##Inference Decoder
        with tf.name_scope('lstm_inference_decoder'):
            inference_stacked_decoder=tf.nn.rnn_cell.MultiRNNCell([get_a_cell(hparam.hidden_dim) for _ in range(hparam.number_of_layers)])

        with tf.variable_scope("decode", reuse=True):
            # Inference Helper
            def initialize_fn():
                finished = tf.tile([False], [hparam.batch_size])
                start_inputs = tf.fill([hparam.batch_size, hparam.out_feature], -1.)
                print("start_inputs___",start_inputs.shape)
                return (finished, start_inputs)

            def sample_fn(time, outputs, state):
                return tf.constant([0])

            def next_inputs_fn(time, outputs, state, sample_ids):
                finished = (time+1)  >= hparam.output_seq_len


                #finished =time >= output_seq_len

                next_inputs = outputs
                print("outputs_______",outputs.shape)

                return (finished, next_inputs, state)


            inference_helper = tf.contrib.seq2seq.CustomHelper(initialize_fn=initialize_fn,
                                  sample_fn=sample_fn,
                                  next_inputs_fn=next_inputs_fn)
            #Attention
            if hparam.use_attention:
                inference_stacked_decoder = tf.contrib.seq2seq.AttentionWrapper(inference_stacked_decoder, attention_mechanism,attention_layer_size=hparam.hidden_dim)
                initial_state = inference_stacked_decoder.zero_state(hparam.batch_size, tf.float32).clone(cell_state=encoder_state)
            else:
                initial_state = encoder_state

            inference_decoder = tf.contrib.seq2seq.BasicDecoder(inference_stacked_decoder, inference_helper, initial_state, output_layer=projection_layer)
            inference_final_outputs, inference_final_state, inference_final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(inference_decoder,output_time_major=hparam.time_major)
            self.inference_logits = inference_final_outputs.rnn_output
            """


          # Train
        self.global_step = tf.Variable(0, name='global_step', trainable=False)


        # Train
        #self.loss =tf.losses.mean_squared_error(labels=self.decoder_targets,logits=self.logits))

         # Train
        with tf.variable_scope('train_Loss'):
            self.loss_ = tf.reduce_mean(tf.nn.l2_loss(self.logits-self.decoder_targets))
            tf.summary.scalar('train_loss', self.loss_)
            #self.inference_loss_ =tf.reduce_mean( tf.nn.l2_loss(self.inference_logits-self.decoder_targets))
            #tf.tf.summary.scalar('inference_loss', self.inference_loss)
            """
            tf.summary.scalar(
                "train_Loss",
                self.loss_
                    )
            tf.summary.scalar(
                "inference_loss",
                self.inference_loss_
                    )
                    # L2 loss"""

            reg_loss = 0
            for tf_var in tf.trainable_variables():
                if not ("Bias" in tf_var.name or "Output_" in tf_var.name):
                    reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))

            tf.summary.scalar('reg_loss',reg_loss)


            self.loss = self.loss_  + hparam.lambda_l2_reg * reg_loss
            #self.inference_loss= self.inference_loss_ + hparam.lambda_l2_reg * reg_loss
            tf.summary.scalar('train_loss_l2', self.loss)
            #tf.summary.scalar('inference_loss_l2', self.inference_loss)


        # Optimization
        optimizer = tf.train.AdamOptimizer(hparam.learning_rate,
                                            beta1=0.9,
                                            beta2=0.999,
                                            epsilon=1e-08,name='Adam')
        #self.train_op=optimizer.minimize(self.loss)
        self.train_op=optimizer.minimize(self.loss, global_step=self.global_step)

        self.saver = tf.train.Saver()
        self.merged = tf.summary.merge_all()

def create_model(sess, arguments):

    model = Model(arguments)
    sess.run(tf.global_variables_initializer())


    #ckpt = tf.train.get_checkpoint_state(arguments.ckpt_dir)

    #if ckpt and ckpt.model_checkpoint_path:
    #print("Restoring old model parameters from %s" %ckpt.model_checkpoint_path)
    model.saver.restore(sess,'./model_20000')
    #else:
    #print("Created new model.")
    return model


def train_inference(sess, model, arguments, input_batch, out_seq):

    #print ("Started Inference")
    #input_batch,out_seq = generate_inference_batches(x_train, y_train, arguments.batch_size, arguments.input_seq_len, arguments.time_major)

    dec_targets = np.zeros((out_seq.shape))
    feed_dict = {
        model.enc_inp:input_batch,
        model.decoder_targets:dec_targets,  #feeding the targets as zeroes
        model.decoder_lengths: np.ones((arguments.batch_size), dtype=int) * arguments.dec_length,
        model.enc_seq_len : np.ones((arguments.batch_size), dtype=int) * arguments.enc_length
      }

    final_preds,inf_loss = sess.run([model.inference_logits,model.inference_loss], feed_dict)
    if type(final_preds) == list:
        final_preds = final_preds[0]
    #print("Start_time" ,start_time)
    #pred_outputs = cp_final_preds.copy()
    return input_batch,out_seq,final_preds,inf_loss

def validation(sess, model, arguments, x_test, y_test):

    #Creating test input and output batches
    test_in, test_out= generate_test_batches(x_test, y_test, arguments.batch_size, arguments.input_seq_len, arguments.output_seq_len, arguments.time_major,arguments.n_in_features,arguments.n_out_feature)
    test_dec_in =test_out.copy()
    test_dec_in =np.zeros((test_dec_in.shape))
    test_tar=test_dec_in

    feed_dict = {
            model.enc_inp:test_in,
            model.decoder_targets:test_tar,  # feeding the targets as zeroes
            model.decoder_lengths: np.ones((arguments.batch_size), dtype=int) * arguments.dec_length,
            model.enc_seq_len : np.ones((arguments.batch_size), dtype=int) * arguments.enc_length
    }

    pred_loss,validation_summary = sess.run([model.loss,model.merged], feed_dict)
    return pred_loss,validation_summary

def train(sess, model, arguments, x_train, y_train, x_test, y_test,scaler,train_writer,test_writer):


    epoch_num=0
    batches_to_comp_all_data = int(arguments.limit/arguments.batch_size)

    train_time=time.time()
    print ("Training the model")
    for t in range(arguments.epochs):

        train_input_batch, train_output_batch, arguments.last_batch_no = generate_train_batches(x_train, y_train, arguments.batch_size, arguments.input_seq_len, arguments.output_seq_len, arguments.time_major, arguments.seq_batch, arguments.last_batch_no,arguments.n_in_features,arguments.n_out_feature)

        #train_input_batch, train_output_batch, arguments.last_batch_no = generate_train_batches(x_train, y_train, arguments.batch_size, arguments.input_seq_len, arguments.output_seq_len, arguments.time_major, arguments.seq_batch, arguments.last_batch_no,arguments.n_features)

        feed_dicts = {
                model.enc_inp:train_input_batch,
                model.decoder_targets:train_output_batch,
                model.decoder_lengths: np.ones((arguments.batch_size), dtype=int) * arguments.dec_length,
                model.enc_seq_len : np.ones((arguments.batch_size), dtype=int) * arguments.enc_length
                     }

        _, loss_value,global_steps,train_summary = sess.run([model.train_op, model.loss,model.global_step,model.merged], feed_dict=feed_dicts)
       # val_loss,validation_summary = validation(sess, model, arguments, x_test, y_test)
        val_loss=None
        #train_loses.append(loss_value)
        if arguments.epochs % batches_to_comp_all_data ==0:
            epoch_num +=1

        #_,_, _,inf_loss,test_summary = inference(sess, model, arguments, x_train, y_train)
        train_writer.add_summary(train_summary, global_steps)
        #test_writer.add_summary(validation_summary, global_steps)
        test_loss=None
        #inf_loss=None
        #valid_test_loses.append(test_loss)
        if t %100 ==0:
            print("Iteration/Total_Iteration :: {}/{} ,global_steps {}   Train_loss_value::{} , Validation_loss {} ,Time :: {}".format(t,arguments.epochs,global_steps,loss_value,val_loss,(time.time() - train_time)))

        if global_steps %1000 ==0   and global_steps >=1000:
            model_name='./model_' + str(global_steps)
            model.saver.save(sess,model_name)
	    '''
            for i in range(0,18):
                global start_time
                start_time =1550217600 +(i*86400)
                arguments.last_ts =[start_time]

                x_train, y_train, x_test, y_test, scaler = load_data(arguments)
                input_batch, out_seq, predictions, inf_loss = inference(sess, model, arguments, x_train, y_train)

                if arguments.time_major:
                    final_batch_output = predictions[:,-1,:]
                else:
                    final_batch_output = predictions[-1,:,:]



                final_batch_output = scaler.inverse_transform(final_batch_output)
             #   write_data(arguments,final_batch_output)
			 
            start_time =1550217600
	    '''




    #model.saver.save(sess, arguments.ckpt_dir)
    model.saver.save(sess, './model')
    print("Checkpoint saved at model_load")
    #start_time=start_time_value
    train_time=time.time() -train_time
    return  train_input_batch,train_output_batch,_,train_time,loss_value,test_loss,global_steps


def inference(sess, model, arguments, x_train, y_train):

    print ("Started Inference")
    input_batch,out_seq,last_batch = generate_inference_batches(x_train, y_train, arguments.batch_size, arguments.input_seq_len, arguments.time_major,arguments.n_in_features,arguments.n_out_feature)
    #print("in", input_batch[:,-1,:] )
    #print("out",out_seq[:,-1,:])
    if arguments.n_in_features != arguments.n_out_feature:
        enc_in = last_batch
        print("singnal are not same input is last batch")
    else:
        enc_in = out_seq
        print("features are equal")



    dec_targets = np.zeros((out_seq.shape))
    feed_dict = {
        model.enc_inp:enc_in,
        model.decoder_targets:dec_targets,  #feeding the targets as zeroes
        model.decoder_lengths: np.ones((arguments.batch_size), dtype=int) * arguments.dec_length,
        model.enc_seq_len : np.ones((arguments.batch_size), dtype=int) * arguments.enc_length,
        #model.keep_prob : 1
      }

    """if arguments.loopback:
        final_preds, inf_loss = sess.run([model.inference_logits,model.inference_loss], feed_dict)
    else:"""
    final_preds, inf_loss = sess.run([model.logits, model.loss], feed_dict)

    if type(final_preds) == list:
        final_preds = final_preds[0]
    return input_batch,out_seq, final_preds, inf_loss


def capture_info(arguments,train_time,loss_value,test_loss):

    f=open("meta_data.txt","a+")

    f.write("read table : %s\n write table :%s \n interval : %d\n limit :%d \n decoder_input_as_target : %r\n attention: %r \n inference: %r \n time major : %r \n seq_batch: %r\ninput seq length:%d\n batch size:%d\n learning rate:%f \n hidden_dim:%d\n epochs:%d \ntrain_time =%d \nloss_value=%f \n\n\n\n"%(arguments.tables[0],arguments.w_measurement[0],arguments.interval,arguments.limit,arguments.decoder_input_as_target,arguments.use_attention,arguments.inference,arguments.time_major,arguments.seq_batch,arguments.input_seq_len,arguments.batch_size,arguments.learning_rate,arguments.hidden_dim,arguments.epochs,train_time,loss_value))

    f.close()

    return

#start_time_value=1535412645 #time.time()  August 13, 2018 3:30:45 AM

start_time =1543622400 #+(i*86400) #time.time() # start_time_value
tf.reset_default_graph()
sess = tf.InteractiveSession()


train_writer = tf.summary.FileWriter("logs/train", sess.graph)
test_writer = tf.summary.FileWriter("logs/test", sess.graph)

arguments = parameters()
model = create_model(sess, arguments)
x_train, y_train, x_test, y_test, scaler = load_data(arguments)
#print("x_train and Y_train" ,x_train[-5:] , "\n" ,y_train[-5:])

train_input_batch,train_output_batch,train_outputs,train_time,loss_value,test_loss,global_steps=train(sess, model, arguments, x_train, y_train, x_test, y_test,scaler,train_writer,test_writer)


for i in range(0,18):
    global strat_time
    start_time =1543622400 +(i*86400) #time.time() # start_time_value
    arguments.last_ts =[start_time]
#    tf.reset_default_graph()
#    sess = tf.InteractiveSession()


#    train_writer = tf.summary.FileWriter("logs/train", sess.graph)
#    test_writer = tf.summary.FileWriter("logs/test", sess.graph)

#    arguments = parameters()
#    model = create_model(sess, arguments)
    x_train, y_train, x_test, y_test, scaler = load_data(arguments)
 #   print("test: " , x_test[:5] , "Y ", y_test[:5])

    #train_input_batch,train_output_batch,train_outputs,train_time,loss_value,test_loss,global_steps=train(sess, model, arguments, x_train, y_train, x_test, y_test,scaler,train_writer,test_writer)


    input_batch, out_seq, predictions, inf_loss = inference(sess, model, arguments, x_train, y_train)

    

    if arguments.time_major:
        final_batch_output = predictions[:,-1,:]
    else:
        final_batch_output = predictions[-1,:,:]



    final_batch_output = scaler.inverse_transform(final_batch_output)
    write_data(arguments,final_batch_output)

#color='g'  # g r b k.d(45k e)
#last_batch_data_ploting(input_batch,predictions,scaler,out_seq,arguments.time_major,color )
#capture_info(arguments,train_time,loss_value,test_loss)

print("Final Loses::: , Train_loss {} , Test_loss {} , Infrence_loss {}, Global_steps  {}".format(loss_value,test_loss,inf_loss,global_steps))


sess.close()

