seq_len    =  n = 10
n_features = 1
state_size = 4

n_output = 2

init_alpha_     = tf.placeholder_with_default(tf.ones([1]),[1],name='init_op_layer') # feed at init or 1
init_beta_      = tf.placeholder_with_default(tf.ones([1]),[1],name='init_op_layer') # feed at init or 1
penalty_parameter_ = tf.placeholder(tf.float32,name='penalty_parameter')
lr_                = tf.placeholder(tf.float32,name='learning_rate')
batch_size_        = tf.placeholder(tf.int32,  name='batch_size')

X_       = tf.placeholder("float32", shape=[seq_len,None,n_features ],name='features')
Y_       = tf.placeholder("float32", shape=[seq_len,None, 1],name='waiting_times')
U_       = tf.placeholder("float32", shape=[seq_len,None, 1],name='censoring_indicators') #
weight_  = tf.placeholder("float32", shape=[seq_len,None, 1],name='weight_') #

early_stop_w_  = tf.placeholder(tf.int32,name = 'early_stop_w')

def simple_lstm(X_,early_stop,batch_size_, name=None):
    with tf.name_scope(name):
        with tf.name_scope(name+"/input_layer"):
            X_flat  = tf.reshape(X_, [-1,n_features])                          #[seq_len*batch_size,n_features]

            hidden1 = create_layer(X_flat,n_features,
                                   state_size,
                                   name=name+'/hidden1', 
                                   activation_function = tf.nn.tanh) #[seq_len*batch_size,state_size]
            hidden1_out = tf.split(0,seq_len, hidden1)                         #seq_len*(batch_size,state_size)

        cell  = rnn_cell.BasicLSTMCell(state_size, forget_bias=1.0)
        #seq_len*(batch_size,state_size), seq_len*(batch_size,2*state_size)
        initial_state = cell.zero_state(batch_size_, tf.float32)
        rnn_outputs, states = rnn.rnn(cell, 
                                      hidden1_out, 
                                      initial_state=initial_state,
                                      scope=name+'/RNN', 
                                      sequence_length=early_stop)

        with tf.name_scope("output_layer%s"%(name)):
            rnn_outputs_nice = tf.pack(rnn_outputs)                       # [seq_len,batch_size,state_size ]
            rnn_outputs_flat = tf.reshape(rnn_outputs_nice, [-1,state_size]) # [seq_len*batch_size,state_size])

            init_op_layer = tf.ones([2]) # DEBUG # Centers softplus at init_op_layer
            op_activation = create_layer(rnn_outputs_flat, 
                                        state_size,
                                        n_output,name='output_layer',
                                        activation_function = tf.identity,
                                        init_biases=init_op_layer
                                        )
    return op_activation


def training_step_rnn(logLik_raw,early_stop_,batch_size_,lr_):
    with tf.name_scope("training"):
        # As we have 'early stop' we need to slice away i.e first early_stop*batch_size after flattening
        # We also need to reshape awkwardly OR simply feed in early_stop_ : [some value]
        # But that's awful.
        logLik_masked = tf.slice(tf.squeeze(logLik_raw), [0], tf.reshape(tf.mul(early_stop_,batch_size_),[1])) 
#        cost = -tf.div(tf.reduce_sum(logLik_masked),tf.mul(seq_len+0.0,tf.to_float(batch_size_)))
        cost = -tf.reduce_mean(logLik_masked) # Varying length seq has same contribution
        step = tf.train.RMSPropOptimizer(learning_rate=lr_,epsilon=1e-5).minimize(cost)
    return logLik_masked, cost, step
        
with tf.name_scope("weibull_lstm"):
    op_activation = simple_lstm(X_,early_stop_w_,batch_size_, name="RNN_weibull")

    a_, b_ = tf.split(1, 2, op_activation)
    
    op_layer_w = tf.pack(a_,b_)

    logLik_w    = weibull_logLik_discrete(a_,b_, tf.reshape(Y_,[-1,1]), tf.reshape(U_,[-1,1]),name="logLikelihood")
    
    _, cost_w, step_w   = training_step_rnn(logLik_w,early_stop_w_,batch_size_,lr_)

    logLik_w_regularized = logLik_w-weibull_beta_wall(b_,location = 8.0, growth=10.0, name='penalty_calculation')
    _, cost_w_pen, pen_step_w  = training_step_rnn(logLik_w_regularized,early_stop_w_,batch_size_,lr_)
    
    _, cost_w_weighted, step_w_weighted = training_step_rnn(logLik_w*tf.reshape(weight_,[-1,1]),early_stop_w_,batch_size_,lr_)
    
sess = tf.InteractiveSession()
summary_writer = tf.train.SummaryWriter("./logs/nn_logs/simple_ex", sess.graph) # for 0.8
summaries = tf.merge_all_summaries()
saver = tf.train.Saver()