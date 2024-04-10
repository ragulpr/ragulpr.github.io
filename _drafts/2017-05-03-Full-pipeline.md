---
layout: post
title: XXXXXXXXXXXXXXXXXX
---

# WTTE-pipeline : data-munging template


Simple pipeline. Take by-timestamp records-data, aggregate over larger timeinterval (discretize) and reshape to tensor. 


# Example pipe

FLOW :
1. Munge df ( want id,time to be unique keys)
2. Decide on what happens between observed timesteps
    * Discrete time:
        * Pad between timesteps (use t_elapsed) or not (use t_ix) 
    * Continuous time:
        * Do not pad between timesteps as 'step' is undefined.
3. Embed in fix-format tensor via rightpadding
    
#### Devils in the details: Continuous, discrete and discretized time is different.

t_elapsed = integer or double time 0,1,19,30,...

t_ix  = integer dense rank of t_elapsed 0,1,2,3,...

t = t_elapsed or t_ix depending on context. Good practice to be specific or keep all of them. Premature optimization is the yadayada

### Meta-example: Tensorflow commit data

    cd tensorflow
    FILENAME="tensorflow.csv"

    echo commit,author_name,time_sec,subject,files_changed,lines_inserted,lines_deleted>../$FILENAME;

    git log --oneline --pretty="_Z_Z_Z_%h_Y_Y_\"%an\"_Y_Y_%at_Y_Y_\"%<(79,trunc)%f\"_Y_Y__X_X_"  --stat    \
        | grep -v \| \
        | sed -E 's/@//g' \
        | sed -E 's/_Z_Z_Z_/@/g' \
        |  tr "\n" " "   \
        |  tr "@" "\n" |sed -E 's/,//g'  \
        | sed -E 's/_Y_Y_/, /g' \
        | sed -E 's/(changed [0-9].*\+\))/,\1,/'  \
        | sed -E 's/(changed [0-9]* deleti.*-\)) /,,\1/' \
        | sed -E 's/insertion.*\+\)//g' \
        | sed -E 's/deletion.*\-\)//g' \
        | sed -E 's/,changed/,/' \
        | sed -E 's/files? ,/,/g'  \
        | sed -E 's/_X_X_ $/,,/g'  \
        | sed -E 's/_X_X_//g' \
        | sed -E 's/ +,/,/g' \
        | sed -E 's/, +/,/g'>>../$FILENAME;
        



```python
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import wtte.tte_util as tte
import wtte.transforms as tr

from IPython import display
```


```python
pd.options.display.max_rows = 10

path = "~/Prylar/clones/logs/"
# filename = "amazon-dsstne.csv"
# filename = "caffe.csv"
# filename = "CNTK.csv"
# filename = "convnetjs.csv"
# filename = "deeplearning4j.csv"
# filename = "h2o-3.csv"
# filename = "incubator-singa.csv"
# filename = "keras.csv"
# filename = "mxnet.csv"
# filename = "Paddle.csv"
filename = "tensorflow.csv"
# filename = "Theano.csv"
# filename = "torch7.csv"
# filename = "veles.csv"

df = pd.read_csv(path+filename,error_bad_lines=False)

df.fillna(0,inplace=True)

# Create a fictitious integer id based on first commit.
# (order has no downstream implications except easy plotting.)
id_col = 'id'
df[id_col] = df.groupby(["author_name"], group_keys=False).\
               apply(lambda g: g.time_sec.min().\
               astype(str)+g.author_name).\
               rank(method='dense').astype(int)

df.plot(kind='scatter', x='time_sec', y='id',s=0.1)
plt.title('commits')
plt.xlabel('time')
plt.ylabel('user id')
plt.show()
    
df.sort_values(['id','time_sec'],inplace=True)

df

```


![png](/assets/data_pipeline_files/data_pipeline_2_0.png)





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>commit</th>
      <th>author_name</th>
      <th>time_sec</th>
      <th>subject</th>
      <th>files_changed</th>
      <th>lines_inserted</th>
      <th>lines_deleted</th>
      <th>id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>16394</th>
      <td>f41959c</td>
      <td>Manjunath Kudlur</td>
      <td>1446856078</td>
      <td>TensorFlow-Initial-commit-of-TensorFlow-librar...</td>
      <td>1900</td>
      <td>391534</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16393</th>
      <td>cd9e60c</td>
      <td>Manjunath Kudlur</td>
      <td>1446863831</td>
      <td>TensorFlow-Upstream-latest-changes-to-Git     ...</td>
      <td>72</td>
      <td>1289</td>
      <td>958</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16387</th>
      <td>71842da</td>
      <td>Manjunath Kudlur</td>
      <td>1447019816</td>
      <td>TensorFlow-Upstream-latest-changes-to-git     ...</td>
      <td>14</td>
      <td>110</td>
      <td>110</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16386</th>
      <td>1d3874f</td>
      <td>Manjunath Kudlur</td>
      <td>1447024477</td>
      <td>TensorFlow-Upstream-changes-to-git            ...</td>
      <td>22</td>
      <td>430</td>
      <td>405</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16385</th>
      <td>b2dc60e</td>
      <td>Manjunath Kudlur</td>
      <td>1447033308</td>
      <td>TensorFlow-Upstream-changes-to-git            ...</td>
      <td>34</td>
      <td>398</td>
      <td>314</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>d53b5a4</td>
      <td>Maxwell Paul Brickner</td>
      <td>1492135693</td>
      <td>Updated-the-link-to-tensorflow.org-to-use-http...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>861</td>
    </tr>
    <tr>
      <th>9</th>
      <td>f83c3da</td>
      <td>cfperez</td>
      <td>1492210377</td>
      <td>Correct-tf.matmul-keyword-reference-in-docs   ...</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>862</td>
    </tr>
    <tr>
      <th>8</th>
      <td>b365526</td>
      <td>Daniel Rasmussen</td>
      <td>1492223657</td>
      <td>Support-int32-idxs-in-sparse_tensor_dense_matm...</td>
      <td>7</td>
      <td>140</td>
      <td>111</td>
      <td>863</td>
    </tr>
    <tr>
      <th>7</th>
      <td>a09eeb2</td>
      <td>wannabesrevenge</td>
      <td>1492311574</td>
      <td>Tensorboard-change-NPM-script-name-prepare-to-...</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>864</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0e17162</td>
      <td>Egil Martinsson</td>
      <td>1492428528</td>
      <td>fix-style_guide.md-my_op-example              ...</td>
      <td>1</td>
      <td>25</td>
      <td>25</td>
      <td>865</td>
    </tr>
  </tbody>
</table>
<p>16395 rows × 8 columns</p>
</div>




```python
# dt = "wallclock time", global timestep i.e 2012-01-01,...
abs_time_col='dt'
# t  = "elapsed time", local timestep i.e 0,1,2,10,...
t_col = 't_elapsed' 
discrete_time = True
# Does the explicit rows in dataset cover each sequence whole life?
sequences_terminated = False
numeric_cols = ["n_commits","files_changed", "lines_inserted","lines_deleted"]

if discrete_time:
    # Convert nanosec to date
    df[abs_time_col] = pd.to_datetime(df['time_sec'],unit='s').dt.date
    # Last timestep may be incomplete/not fully measured so drop it.
    df = df.loc[df[abs_time_col] <= df[abs_time_col].max()]
else:
    # Left as human readable format for readability.
#    df[abs_time_col] = pd.to_datetime(df['time_sec'],unit='s')
    df[abs_time_col] = df['time_sec']

# here we have the special case that a row indicates an event:
df['n_commits'] = 1

# Aggregate over the new datetime interval to get id,dt = unique key value pair
df = df.groupby([id_col,'author_name',abs_time_col],as_index=False).\
    agg(dict.fromkeys(numeric_cols, "sum"))#.reset_index()

# event = if something special happened i.e commit.
df['event'] =  (df.n_commits>0).astype(int)

if not sequences_terminated:
    # Assuming each sequence has its own start and is not terminated by last event:
    # Add last time that we knew the sequence was 'alive'.
    df = tr.df_join_in_endtime(df,
               per_id_cols=[id_col,'author_name'], 
               abs_time_col=abs_time_col,nanfill_val = 0)
    # Warning: fills unseen timesteps with 0
    
# Add "elapsed time" t_elapsed = 0,3,99,179,.. for each user. 
df[t_col] = df.groupby([id_col], group_keys=False).apply(lambda g: g.dt-g.dt.min())

if discrete_time:
    # Make it a well-behaved integer:
    # infer the discrete stepsize as the first component of the timedelta
    df[t_col] = df[t_col].dt.components.ix[:,0] 
else:
    # Add t_ix = 0,1,2,3,.. and set as primary user-time indicator.
    # if we pass t_elapsed as t_col downstream we'll pad between observed secs
    df['t_ix'] = df.groupby([id_col])[t_col].rank(method='dense').astype(int)-1
    t_col = 't_ix'

df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>author_name</th>
      <th>dt</th>
      <th>files_changed</th>
      <th>n_commits</th>
      <th>lines_deleted</th>
      <th>lines_inserted</th>
      <th>event</th>
      <th>t_elapsed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>865</th>
      <td>1</td>
      <td>Manjunath Kudlur</td>
      <td>2015-11-07</td>
      <td>1972</td>
      <td>2</td>
      <td>958</td>
      <td>392823</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>866</th>
      <td>1</td>
      <td>Manjunath Kudlur</td>
      <td>2015-11-08</td>
      <td>36</td>
      <td>2</td>
      <td>515</td>
      <td>540</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>867</th>
      <td>1</td>
      <td>Manjunath Kudlur</td>
      <td>2015-11-09</td>
      <td>68</td>
      <td>7</td>
      <td>1755</td>
      <td>1888</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>868</th>
      <td>1</td>
      <td>Manjunath Kudlur</td>
      <td>2015-11-20</td>
      <td>1271</td>
      <td>1</td>
      <td>1067</td>
      <td>18402</td>
      <td>1</td>
      <td>13</td>
    </tr>
    <tr>
      <th>869</th>
      <td>1</td>
      <td>Manjunath Kudlur</td>
      <td>2015-11-25</td>
      <td>322</td>
      <td>3</td>
      <td>2888</td>
      <td>5183</td>
      <td>1</td>
      <td>18</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>7506</th>
      <td>863</td>
      <td>Daniel Rasmussen</td>
      <td>2017-04-15</td>
      <td>7</td>
      <td>1</td>
      <td>111</td>
      <td>140</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>862</th>
      <td>863</td>
      <td>Daniel Rasmussen</td>
      <td>2017-04-17</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7507</th>
      <td>864</td>
      <td>wannabesrevenge</td>
      <td>2017-04-16</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>863</th>
      <td>864</td>
      <td>wannabesrevenge</td>
      <td>2017-04-17</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>864</th>
      <td>865</td>
      <td>Egil Martinsson</td>
      <td>2017-04-17</td>
      <td>1</td>
      <td>1</td>
      <td>25</td>
      <td>25</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>7508 rows × 9 columns</p>
</div>




```python
# Currently haven't found right paramtuning for ultra-high censoring.
# until then, let's focus on those who committed at least 10 days:
df['n_events_total']= df.groupby([id_col], group_keys=False).apply(lambda g: g.event.sum()+g.event-g.event)
df = df.loc[df['n_events_total'] > 10]

df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>author_name</th>
      <th>dt</th>
      <th>files_changed</th>
      <th>n_commits</th>
      <th>lines_deleted</th>
      <th>lines_inserted</th>
      <th>event</th>
      <th>t_elapsed</th>
      <th>n_events_total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>865</th>
      <td>1</td>
      <td>Manjunath Kudlur</td>
      <td>2015-11-07</td>
      <td>1972</td>
      <td>2</td>
      <td>958</td>
      <td>392823</td>
      <td>1</td>
      <td>0</td>
      <td>91</td>
    </tr>
    <tr>
      <th>866</th>
      <td>1</td>
      <td>Manjunath Kudlur</td>
      <td>2015-11-08</td>
      <td>36</td>
      <td>2</td>
      <td>515</td>
      <td>540</td>
      <td>1</td>
      <td>1</td>
      <td>91</td>
    </tr>
    <tr>
      <th>867</th>
      <td>1</td>
      <td>Manjunath Kudlur</td>
      <td>2015-11-09</td>
      <td>68</td>
      <td>7</td>
      <td>1755</td>
      <td>1888</td>
      <td>1</td>
      <td>2</td>
      <td>91</td>
    </tr>
    <tr>
      <th>868</th>
      <td>1</td>
      <td>Manjunath Kudlur</td>
      <td>2015-11-20</td>
      <td>1271</td>
      <td>1</td>
      <td>1067</td>
      <td>18402</td>
      <td>1</td>
      <td>13</td>
      <td>91</td>
    </tr>
    <tr>
      <th>869</th>
      <td>1</td>
      <td>Manjunath Kudlur</td>
      <td>2015-11-25</td>
      <td>322</td>
      <td>3</td>
      <td>2888</td>
      <td>5183</td>
      <td>1</td>
      <td>18</td>
      <td>91</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>7377</th>
      <td>766</td>
      <td>Yong Tang</td>
      <td>2017-03-23</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>26</td>
      <td>1</td>
      <td>19</td>
      <td>11</td>
    </tr>
    <tr>
      <th>7378</th>
      <td>766</td>
      <td>Yong Tang</td>
      <td>2017-03-25</td>
      <td>1</td>
      <td>1</td>
      <td>55</td>
      <td>7</td>
      <td>1</td>
      <td>21</td>
      <td>11</td>
    </tr>
    <tr>
      <th>7379</th>
      <td>766</td>
      <td>Yong Tang</td>
      <td>2017-03-26</td>
      <td>1</td>
      <td>1</td>
      <td>9</td>
      <td>9</td>
      <td>1</td>
      <td>22</td>
      <td>11</td>
    </tr>
    <tr>
      <th>7380</th>
      <td>766</td>
      <td>Yong Tang</td>
      <td>2017-04-10</td>
      <td>3</td>
      <td>1</td>
      <td>5</td>
      <td>14</td>
      <td>1</td>
      <td>37</td>
      <td>11</td>
    </tr>
    <tr>
      <th>765</th>
      <td>766</td>
      <td>Yong Tang</td>
      <td>2017-04-17</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>44</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
<p>5306 rows × 10 columns</p>
</div>




```python
def _align_padded(padded,align_right):
    """aligns nan-padded temporal arrays to the right (align_right=True) or left.
    """
    padded = np.copy(padded)
    
    if len(padded.shape)==2:
        # (n_seqs,n_timesteps)
        seq_lengths = (False == np.isnan(padded)).sum(1)
        is_flat = True
        padded = np.expand_dims(padded,-1)
    elif len(padded.shape)==3:
        # (n_seqs,n_timesteps,n_features,..)
        seq_lengths = (False == np.isnan(padded[:,:,0])).sum(1)
        is_flat =False
    else:
        print 'not yet implemented'
        # TODO 
        
    n_seqs = padded.shape[0]
    n_timesteps = padded.shape[1]

    if align_right:
        for i in xrange(n_seqs):
            n = seq_lengths[i]
            if n>0:
                padded[i,(n_timesteps-n):,:] = padded[i,:n,:]
                padded[i,:(n_timesteps-n),:] = np.nan
    else:
        for i in xrange(n_seqs):
            n = seq_lengths[i]
            if n>0:
                padded[i,:n,:] = padded[i,(n_timesteps-n):,:]
                padded[i,n:,:] = np.nan
                        
    if is_flat:
        padded = np.squeeze(padded)
        
    return padded

def right_pad_to_left_pad(padded):
    return _align_padded(padded,align_right= True)

def left_pad_to_right_pad(padded):
    return _align_padded(padded,align_right= False)

# padded = tr.df_to_padded(df,["n_commits"],t_col=t_col)
# np.testing.assert_array_equal(padded,left_pad_to_right_pad(right_pad_to_left_pad(padded)))

# padded = np.copy(np.squeeze(padded))

# np.testing.assert_array_equal(padded,left_pad_to_right_pad(right_pad_to_left_pad(padded)))

```

# To trainable Tensor format
* map dataframes to tensor
* calculate tte-values
* Split into train/test
* normalize (using training data)
* hide truth from the model. Very important, otherwise it causes NaNs



```python
# feature_cols = ["n_commits","files_changed", "lines_inserted","lines_deleted"]
feature_cols = ["n_commits"]
x = tr.df_to_padded(df,feature_cols,t_col=t_col)
events = tr.df_to_padded(df,['event'],t_col=t_col).squeeze() # For tte/censoring calculation 

n_timesteps = x.shape[1]
n_features  = x.shape[-1]
n_sequences = x.shape[0]
seq_lengths = (False==np.isnan(x[:,:,0])).sum(1)

x = np.log(x+1.) # log-kill outliers, decent method since we have positive data

if discrete_time:
    padded_t = None
else:
    padded_t = tr.df_to_padded(df,['t_elapsed'],t_col=t_col).squeeze()

y = np.zeros([n_sequences,n_timesteps,2])

# # Sort by seq length for pretty plots (OPTIONAL)
# new_ix = np.argsort(seq_lengths)
# x = x[new_ix,:,:]
# events = events[new_ix,:]
# seq_lengths = seq_lengths[new_ix]

# SPLIT 
# Simplest way to setup cross validation is to hide the t last days. bonus result is less censoring
# brutal method: simply right align all tensors and simply cut off the last 10% of timesteps. (ex last 30 days)

n_timesteps_to_hide = np.floor(n_timesteps*0.1).astype(int)

x_train      = left_pad_to_right_pad(right_pad_to_left_pad(x)[:,:(n_timesteps-n_timesteps_to_hide),:])
y_train      = left_pad_to_right_pad(right_pad_to_left_pad(y)[:,:(n_timesteps-n_timesteps_to_hide),:])
events_train = left_pad_to_right_pad(right_pad_to_left_pad(events)[:,:(n_timesteps-n_timesteps_to_hide)])

n_train     = x_train.shape[0]
seq_lengths_train = (False==np.isnan(x_train[:,:,0])).sum(1)

# Calculate TTE/censoring indicators after split.
y_train[:,:,0] = tr.padded_events_to_tte(events_train,discrete_time=discrete_time,t_elapsed=padded_t)
y_train[:,:,1] = tr.padded_events_to_not_censored(events_train,discrete_time)

y[:,:,0] = tr.padded_events_to_tte(events,discrete_time=discrete_time,t_elapsed=padded_t)
y[:,:,1] = tr.padded_events_to_not_censored(events,discrete_time)


#del x_, y_,seq_lengths

# NORMALIZE
x_train,means,stds = tr.normalize_padded(x_train)
x,_,_         = tr.normalize_padded(x,means,stds)

# HIDE the truth from the model:
if discrete_time:
    x = tr.shift_discrete_padded_features(x)
    x_train = tr.shift_discrete_padded_features(x_train)

    
# Used for initialization of alpha-bias:
tte_mean_train = np.nanmean(y_train[:,:,0])

print 'x_',x.shape,x.dtype
print 'y_',y.shape,y.dtype
print 'x_train',x_train.shape,x_train.dtype
print 'y_train',y_train.shape,y_train.dtype
print 'tte_mean_train: ', tte_mean_train
```

    x_ (92, 528, 1) float64
    y_ (92, 528, 2) float64
    x_train (92, 476, 1) float64
    y_train (92, 476, 2) float64
    tte_mean_train:  23.1710377145



```python
def timeline_plot(padded,title,cmap=None,plot=True):
    fig, ax = plt.subplots(ncols=2, sharey=True,figsize=(12,4))
    
    ax[0].imshow(padded,interpolation='none', aspect='auto',cmap=cmap,origin='lower')    
    ax[0].set_ylabel('sequence');
    ax[0].set_xlabel('sequence time');
        
    im = ax[1].imshow(right_pad_to_left_pad(padded),interpolation='none', aspect='auto',cmap=cmap,origin='lower')  
    ax[1].set_ylabel('sequence');
    ax[1].set_xlabel('absolute time'); #(Assuming sequences end today)
    
    fig.suptitle(title,fontsize=14)
    if plot:
        fig.show()
        return None,None
    else:
        return fig,ax

def timeline_aggregate_plot(padded,title,plot=True):
    fig, ax = plt.subplots(ncols=2,nrows=2,sharex=True, sharey=False,figsize=(12,8))
    
    ax[0,0].plot(np.nanmean(padded,axis=0),lw=0.5,c='black',drawstyle='steps-post')
    ax[0,0].set_title('mean/timestep')
    ax[1,0].plot(np.nansum(padded,axis=0),lw=0.5,c='black',drawstyle='steps-post')
    ax[1,0].set_title('sum/timestep')
    padded = right_pad_to_left_pad(events)   
    ax[0,1].plot(np.nanmean(padded,axis=0),lw=0.5,c='black',drawstyle='steps-post')
    ax[0,1].set_title('mean/timestep')
    ax[0,1].axvline(x=n_timesteps-n_timesteps_to_hide,c="red",linewidth=1,zorder=10)
    ax[1,1].plot(np.nansum(padded,axis=0),lw=0.5,c='black',drawstyle='steps-post')
    ax[1,1].set_title('sum/timestep')

    fig.suptitle(title,fontsize=14)
    if plot:
        fig.show()
        return None,None
    else:
        return fig,ax
    
fig,ax = timeline_plot(events,"events (train left of red line)",cmap="Greys",plot=False)
ax[0].plot(seq_lengths_train,xrange(n_sequences),c="red",linewidth=2,zorder=10,drawstyle = 'steps-post')
ax[1].axvline(x=n_timesteps-n_timesteps_to_hide,c="red",linewidth=2,zorder=10)
plt.show()
del fig,ax

fig, ax = timeline_aggregate_plot(events,'events (aggregate)',plot=False)
ax[1,1].axvline(x=n_timesteps-n_timesteps_to_hide,c="red",linewidth=1,zorder=10)
plt.show()
del fig,ax

print 'TRAINING SET'
train_mask = (False==np.isnan(y_train[:,:,0]))

plt.hist(seq_lengths_train)
plt.title('Distribution of sequence lengths (training set)')
plt.xlabel('sequence length')
plt.show()

plt.hist(y_train[:,:,0][train_mask].flatten(),100)
plt.title('Distribution of censored tte')
plt.ylabel('sequence')
plt.xlabel('t')
plt.show()

plt.hist(y_train[:,:,1][train_mask].flatten(),2)
plt.title('Distribution of censored/non censored points')
plt.xlabel("u")
plt.show()

print '########## features'
for f in xrange(n_features):
    timeline_plot(x_train[:,:,f],feature_cols[f])
    plt.show()
    tmp = x_train[:,:,f].flatten()
    plt.hist(tmp[False==np.isnan(tmp)],10)
    plt.title('Distribution of log '+feature_cols[f])
    plt.show()
print '########## '
del tmp,train_mask

plt.plot(y_train[0,:,0])
plt.title('example seq tte')
plt.xlabel('t')
plt.show()

timeline_plot(y_train[:,:,1]-y_train[:,:,1],'where we have data')
######
timeline_plot(1-y_train[:,:,1],'censoring',cmap='Greys')
timeline_plot(1-y_train[:,:,0],'TTE (censored)',cmap='Greys')

```


![png](/assets/data_pipeline_files/data_pipeline_8_0.png)



![png](/assets/data_pipeline_files/data_pipeline_8_1.png)


    TRAINING SET



![png](/assets/data_pipeline_files/data_pipeline_8_3.png)



![png](/assets/data_pipeline_files/data_pipeline_8_4.png)



![png](/assets/data_pipeline_files/data_pipeline_8_5.png)


    ########## features


    /usr/local/lib/python2.7/site-packages/matplotlib/figure.py:397: UserWarning: matplotlib is currently using a non-GUI backend, so cannot show the figure
      "matplotlib is currently using a non-GUI backend, "



![png](/assets/data_pipeline_files/data_pipeline_8_8.png)



![png](/assets/data_pipeline_files/data_pipeline_8_9.png)


    ########## 



![png](/assets/data_pipeline_files/data_pipeline_8_11.png)





    (None, None)




![png](/assets/data_pipeline_files/data_pipeline_8_13.png)



![png](/assets/data_pipeline_files/data_pipeline_8_14.png)



![png](/assets/data_pipeline_files/data_pipeline_8_15.png)


# Train a WTTE-RNN
Everything done above could be done as one function get_data(). I kept it "in main" for transparency. Clearly this saves you time when you want to cut away the junk.
## TODO
    * Too much censoring causes NaNs (ex dataset: Theano repo)
    * Add masking-support for loss function to train on varying length batches
    * Find tuning to be able to use [-inf,inf] and non-centered activation functions like ReLu without causing NaNs
    * Is it useful to clip Alpha?
    * ... and everything else like best architecture, better output activation function etc.


```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from keras import backend as K

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,GRU
from keras.layers import Lambda
from keras.layers import Dropout
from keras.layers.wrappers import TimeDistributed

import keras.initializers as initializers
from keras.optimizers import RMSprop,adam

from keras.callbacks import History 
from keras.models import load_model

import wtte.tte_util as tte
import wtte.weibull as weibull
import wtte.wtte as wtte

```

    Using TensorFlow backend.


### Some thoughts on architecture:

* Output layer activation is everything. Initialize preferably as alpha=expected val, beta=1
* Bounded activations (e.g tanh) are more stable in short run but creates huge parameters and interfers with the recurrent counting process.
    * TODO get non-bounded (e.g relu) to work without causing NaN
* Think simple. Always try with pure ANN `model.add(TimeDistributed(Dense(...` first!
* Overfitting can be OK if we want to condense the history for analysis 



```python
init_alpha = -1.0/np.log(1.0-1.0/(tte_mean_train+1.0) )
init_alpha = init_alpha/np.nanmean(y_train[:,:,1])
print 'init_alpha: ',init_alpha

np.random.seed(1)

model = Sequential()

model.add(LSTM(10, activation='tanh', input_shape=(None, n_features),return_sequences=True))
model.add(TimeDistributed(Dense(10,activation='tanh')))

model.add(Dense(2))
model.add(Lambda(wtte.output_lambda, arguments={"init_alpha":init_alpha, 
                                               "max_beta_value":4.0}))


loss = wtte.loss(kind='discrete',use_censoring=True).loss_function

lr = 0.1
model.compile(loss=loss, optimizer=adam(lr=lr))

model.summary()

loss_train = []
alpha_means = []
beta_means =  []
epoch_ok = True

def epoch():
    epoch_order = np.random.permutation(n_train)
    x_ = np.copy(np.expand_dims(x_train[epoch_order,:,:], axis=0))
    y_ = np.copy(np.expand_dims(y_train[epoch_order,:,:], axis=0)) 
    seq_lengths_ = np.copy(seq_lengths_train[epoch_order])

    predicted = model.predict(x_[0,:,:,:])
    predicted = predicted + np.expand_dims(predicted[:,:,0],-1)*0 # nan-mask
    alpha_means.append(np.nanmean(predicted[:,:,0]))
    beta_means.append(np.nanmean(predicted[:,:,1]))
    
    plt.scatter(predicted[:,:,0],predicted[:,:,1],s=0.1)
    plt.title('Predicted params')
    plt.xlim([np.nanmin(predicted[:,:,0]),np.nanmax(predicted[:,:,0])])
    plt.ylim([np.nanmin(predicted[:,:,1]),np.nanmax(predicted[:,:,1])])
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$\beta$')
    plt.show()

    for i in xrange(n_train):
        this_seq_length = seq_lengths_[i]
        if this_seq_length>0:
            this_loss = model.train_on_batch(x_[:,i,:this_seq_length,:], 
                                             y_[:,i,:this_seq_length,:])

            loss_train.append(this_loss)
            
            if np.isnan(this_loss):
                print 'induced NANs on iteration ',len(loss_train)
                return False
        

    return True
```

    init_alpha:  27.3083557191
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    lstm_2 (LSTM)                (None, None, 10)          480       
    _________________________________________________________________
    time_distributed_2 (TimeDist (None, None, 10)          110       
    _________________________________________________________________
    dense_4 (Dense)              (None, None, 2)           22        
    _________________________________________________________________
    lambda_2 (Lambda)            (None, None, 2)           0         
    =================================================================
    Total params: 612.0
    Trainable params: 612.0
    Non-trainable params: 0.0
    _________________________________________________________________



```python
#model.load_weights('./model_checkpoint.h5', by_name=False)
```


```python
# store/load models since we may encounter NaNs or training interrupted.
for i in xrange(25):
    lr = 0.01#lr*0.9
    K.set_value(model.optimizer.lr, lr)

    print 'epoch ',i,' training step ',len(loss_train),' lr ',K.eval(model.optimizer.lr)
    if i%5==0 and epoch_ok:
        predicted = model.predict(x_train)
        predicted[:,:,1]=predicted[:,:,1]+predicted[:,:,0]*0# lazy re-add NAN-mask
        expected_val = weibull.mean(predicted[:,:,0], predicted[:,:,1])
        plt.imshow(expected_val,aspect='auto',interpolation="none",origin='lower')  
        plt.colorbar()
        plt.title('predicted (expected value)')
        plt.show()
        
        model.save_weights('./model_checkpoint.h5')  # creates a HDF5 file 'my_model.h5'
        print 'MODEL CHECKPOINT SAVED'
    
    epoch_ok = epoch()
    if epoch_ok:
        n = min([n_train,len(loss_train)])
        print 'alpha mean ',np.mean(alpha_means[-n:]),'beta mean ',np.mean(beta_means[-n:])
    else:
        model.load_weights('./model_checkpoint.h5', by_name=False)
        print 'RESTORING MODEL FROM CHECKPOINT'

```

    epoch  0  training step  0  lr  0.01



![png](/assets/data_pipeline_files/data_pipeline_14_1.png)


    MODEL CHECKPOINT SAVED



![png](/assets/data_pipeline_files/data_pipeline_14_3.png)


    alpha mean  28.0618 beta mean  1.74034
    epoch  1  training step  91  lr  0.01



![png](/assets/data_pipeline_files/data_pipeline_14_5.png)


    alpha mean  27.0033 beta mean  1.18871
    epoch  2  training step  182  lr  0.01



![png](/assets/data_pipeline_files/data_pipeline_14_7.png)


    alpha mean  22.5818 beta mean  1.01351
    epoch  3  training step  273  lr  0.01



![png](/assets/data_pipeline_files/data_pipeline_14_9.png)


    alpha mean  23.4565 beta mean  0.92939
    epoch  4  training step  364  lr  0.01



![png](/assets/data_pipeline_files/data_pipeline_14_11.png)


    alpha mean  22.8956 beta mean  0.868975
    epoch  5  training step  455  lr  0.01



![png](/assets/data_pipeline_files/data_pipeline_14_13.png)


    MODEL CHECKPOINT SAVED



![png](/assets/data_pipeline_files/data_pipeline_14_15.png)


    alpha mean  21.2295 beta mean  0.828921
    epoch  6  training step  546  lr  0.01



![png](/assets/data_pipeline_files/data_pipeline_14_17.png)


    alpha mean  21.9375 beta mean  0.799017
    epoch  7  training step  637  lr  0.01



![png](/assets/data_pipeline_files/data_pipeline_14_19.png)


    alpha mean  22.2791 beta mean  0.782891
    epoch  8  training step  728  lr  0.01



![png](/assets/data_pipeline_files/data_pipeline_14_21.png)


    alpha mean  22.5241 beta mean  0.768547
    epoch  9  training step  819  lr  0.01



![png](/assets/data_pipeline_files/data_pipeline_14_23.png)


    alpha mean  23.1476 beta mean  0.758147
    epoch  10  training step  910  lr  0.01



![png](/assets/data_pipeline_files/data_pipeline_14_25.png)


    MODEL CHECKPOINT SAVED



![png](/assets/data_pipeline_files/data_pipeline_14_27.png)


    alpha mean  23.7855 beta mean  0.752553
    epoch  11  training step  1001  lr  0.01



![png](/assets/data_pipeline_files/data_pipeline_14_29.png)


    alpha mean  24.6724 beta mean  0.746905
    epoch  12  training step  1092  lr  0.01



![png](/assets/data_pipeline_files/data_pipeline_14_31.png)


    alpha mean  24.9946 beta mean  0.735426
    epoch  13  training step  1183  lr  0.01



![png](/assets/data_pipeline_files/data_pipeline_14_33.png)


    alpha mean  25.2241 beta mean  0.73062
    epoch  14  training step  1274  lr  0.01



![png](/assets/data_pipeline_files/data_pipeline_14_35.png)


    alpha mean  25.1851 beta mean  0.726736
    epoch  15  training step  1365  lr  0.01



![png](/assets/data_pipeline_files/data_pipeline_14_37.png)


    MODEL CHECKPOINT SAVED



![png](/assets/data_pipeline_files/data_pipeline_14_39.png)


    alpha mean  25.5915 beta mean  0.722206
    epoch  16  training step  1456  lr  0.01



![png](/assets/data_pipeline_files/data_pipeline_14_41.png)


    alpha mean  25.5898 beta mean  0.719998
    epoch  17  training step  1547  lr  0.01



![png](/assets/data_pipeline_files/data_pipeline_14_43.png)


    alpha mean  25.472 beta mean  0.717262
    epoch  18  training step  1638  lr  0.01



![png](/assets/data_pipeline_files/data_pipeline_14_45.png)


    alpha mean  25.4011 beta mean  0.716342
    epoch  19  training step  1729  lr  0.01



![png](/assets/data_pipeline_files/data_pipeline_14_47.png)


    alpha mean  25.288 beta mean  0.713263
    epoch  20  training step  1820  lr  0.01



![png](/assets/data_pipeline_files/data_pipeline_14_49.png)


    MODEL CHECKPOINT SAVED



![png](/assets/data_pipeline_files/data_pipeline_14_51.png)


    alpha mean  26.279 beta mean  0.711247
    epoch  21  training step  1911  lr  0.01



![png](/assets/data_pipeline_files/data_pipeline_14_53.png)


    alpha mean  26.3265 beta mean  0.709863
    epoch  22  training step  2002  lr  0.01



![png](/assets/data_pipeline_files/data_pipeline_14_55.png)


    alpha mean  26.1908 beta mean  0.711307
    epoch  23  training step  2093  lr  0.01



![png](/assets/data_pipeline_files/data_pipeline_14_57.png)


    alpha mean  26.496 beta mean  0.708793
    epoch  24  training step  2184  lr  0.01



![png](/assets/data_pipeline_files/data_pipeline_14_59.png)


    alpha mean  26.5242 beta mean  0.706323



```python
# Per batch
n_roll_steps = 200
offset = 200
plt.plot(loss_train[offset:], label='training loss')
plt.plot(tte.roll_fun(x=loss_train,fun=np.mean,size=n_roll_steps)[offset:],color='k',lw=3,label='rollmean '+str(n_roll_steps))
plt.xlabel('batch')
plt.legend()
plt.show() 

#plt.plot(loss_train[offset:], label='training loss')
plt.plot(tte.roll_fun(x=loss_train,fun=np.mean,size=n_roll_steps)[offset:],color='k',lw=3,label='rollmean '+str(n_roll_steps))
plt.xlabel('batch')
plt.legend()
plt.show() 

# Per epoch
n_roll_steps = 4
offset = 4
plt.plot(alpha_means[offset:], label='mean alpha')
plt.plot(tte.roll_fun(x=alpha_means,fun=np.mean,size=n_roll_steps)[offset:],color='k',lw=3,label='rollmean '+str(n_roll_steps))
plt.legend()
plt.xlabel('epoch')
plt.show()

plt.plot(beta_means[offset:], label='mean beta')
plt.plot(tte.roll_fun(x=beta_means,fun=np.mean,size=n_roll_steps)[offset:],color='k',lw=3,label='rollmean '+str(n_roll_steps))
plt.legend()
plt.xlabel('epoch')
plt.show()

```


![png](/assets/data_pipeline_files/data_pipeline_15_0.png)



![png](/assets/data_pipeline_files/data_pipeline_15_1.png)



![png](/assets/data_pipeline_files/data_pipeline_15_2.png)



![png](/assets/data_pipeline_files/data_pipeline_15_3.png)


## Predict 


```python
predicted = model.predict(x)
predicted[:,:,1]=predicted[:,:,1]+predicted[:,:,0]*0# lazy re-add NAN-mask
print(predicted.shape)

# Here you'd stop after transforming to dataframe and piping it back to some database
tr.padded_to_df(predicted,column_names=["alpha","beta"],dtypes=[float,float],ids=pd.unique(df.id))
```

    (92, 528, 2)





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>t</th>
      <th>alpha</th>
      <th>beta</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>20.403198</td>
      <td>0.649324</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>1.960579</td>
      <td>0.530411</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2</td>
      <td>1.629222</td>
      <td>0.540398</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>3</td>
      <td>0.872739</td>
      <td>0.593520</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>4</td>
      <td>2.567461</td>
      <td>0.665678</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>31324</th>
      <td>766</td>
      <td>41</td>
      <td>11.467376</td>
      <td>0.645145</td>
    </tr>
    <tr>
      <th>31325</th>
      <td>766</td>
      <td>42</td>
      <td>13.383512</td>
      <td>0.653467</td>
    </tr>
    <tr>
      <th>31326</th>
      <td>766</td>
      <td>43</td>
      <td>15.367269</td>
      <td>0.661231</td>
    </tr>
    <tr>
      <th>31327</th>
      <td>766</td>
      <td>44</td>
      <td>17.414772</td>
      <td>0.667357</td>
    </tr>
    <tr>
      <th>31328</th>
      <td>766</td>
      <td>45</td>
      <td>19.492905</td>
      <td>0.672110</td>
    </tr>
  </tbody>
</table>
<p>31329 rows × 4 columns</p>
</div>



But we keep on plotting
# Individual sequence


```python
drawstyle = 'steps-post'

batch_indx =0 # Guido
#batch_indx =n_train/2

this_seq_len = seq_lengths[batch_indx]
a = predicted[batch_indx,:this_seq_len,0]
b = predicted[batch_indx,:this_seq_len,1]
t = np.array(xrange(len(a)))
x_this = x[batch_indx,:this_seq_len,:]

this_tte = y[batch_indx,:this_seq_len,0]
u = y[batch_indx,:this_seq_len,1]>0

plt.plot(a,drawstyle='steps-post')
plt.title('predicted alpha')
plt.show()
plt.plot(b,drawstyle='steps-post')
plt.title('predicted beta')
plt.show()

fig,ax = plt.subplots()
ax.imshow(x_this.T,origin='lower',interpolation='none',aspect='auto')
ax.set_ylabel('Feature #')
ax.set_title('Features')
ax.set_ylim(-0.5, x_this.shape[-1]-0.5)
ax.set_yticklabels(xrange(x_this.shape[-1]))
ax.set_xlim(-0.5, x_this.shape[0]-0.5)
plt.show()

plt.plot(this_tte,label='censored tte',color='black',linestyle='dashed',linewidth=2,drawstyle=drawstyle)
plt.plot(t[u],this_tte[u],label='uncensored tte',color='black',linestyle='solid',linewidth=2,drawstyle=drawstyle)

plt.plot(weibull.quantiles(a,b,0.75),color='blue',label='pred <0.75',drawstyle=drawstyle)
plt.plot(weibull.mode(a, b), color='red',linewidth=1,label='pred mode/peak prob',drawstyle=drawstyle)
plt.plot(weibull.mean(a, b), color='green',linewidth=1,label='pred mean',drawstyle='steps-post')
plt.plot(weibull.quantiles(a,b,0.25),color='blue',label='pred <0.25',drawstyle=drawstyle)
plt.xlabel('time')
plt.ylabel('time to event')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
```


![png](/assets/data_pipeline_files/data_pipeline_19_0.png)



![png](/assets/data_pipeline_files/data_pipeline_19_1.png)



![png](/assets/data_pipeline_files/data_pipeline_19_2.png)



![png](/assets/data_pipeline_files/data_pipeline_19_3.png)


# Aggregate view


```python
plt.imshow(y_train[:,:,0],aspect='auto',interpolation="none",origin='lower')  
plt.title('tte')
plt.colorbar()
plt.show()

plt.imshow(x.sum(axis=-1),aspect='auto',interpolation="none",origin='lower')  
plt.title('x mean features')
plt.colorbar()
plt.show()

plt.imshow(predicted[:,:,0],aspect='auto',interpolation="none",origin='lower')  
plt.title('alpha')
plt.colorbar()
plt.show()
plt.imshow(predicted[:,:,1],aspect='auto',interpolation="none",origin='lower')  
plt.title('beta')
plt.colorbar()
plt.show()

fig,ax = timeline_plot(y[:,:,0],"tte",plot=False)
ax[0].plot(seq_lengths_train,xrange(n_sequences),c="red",linewidth=2,zorder=10,drawstyle = 'steps-post')
ax[1].axvline(x=n_timesteps-n_timesteps_to_hide,c="red",linewidth=2,zorder=10)
plt.show()

padded = weibull.mean(a=predicted[:,:,0],b=predicted[:,:,1])
fig,ax = timeline_plot(padded,"predicted (expected value)",plot=False)
ax[0].plot(seq_lengths_train,xrange(n_sequences),c="red",linewidth=2,zorder=10,drawstyle = 'steps-post')
ax[1].axvline(x=n_timesteps-n_timesteps_to_hide,c="red",linewidth=2,zorder=10)
plt.show()

fig,ax = timeline_plot(predicted[:,:,0],"alpha",plot=False)
ax[0].plot(seq_lengths_train,xrange(n_sequences),c="red",linewidth=2,zorder=10,drawstyle = 'steps-post')
ax[1].axvline(x=n_timesteps-n_timesteps_to_hide,c="red",linewidth=2,zorder=10)
plt.show()

fig,ax = timeline_plot(predicted[:,:,1],"beta",plot=False)
ax[0].plot(seq_lengths_train,xrange(n_sequences),c="red",linewidth=2,zorder=10,drawstyle = 'steps-post')
ax[1].axvline(x=n_timesteps-n_timesteps_to_hide,c="red",linewidth=2,zorder=10)
plt.show()

padded = -weibull.discrete_loglik(a=predicted[:,:,0],b=predicted[:,:,1],t=y[:,:,0],u=y[:,:,1],equality=False)
fig,ax = timeline_plot(padded,"error",plot=False)
ax[0].plot(seq_lengths_train,xrange(n_sequences),c="red",linewidth=2,zorder=10,drawstyle = 'steps-post')
ax[1].axvline(x=n_timesteps-n_timesteps_to_hide,c="red",linewidth=2,zorder=10)
plt.show()
    
fig, ax = timeline_aggregate_plot(padded,'error (aggregate)',plot=False)
ax[1,1].axvline(x=n_timesteps-n_timesteps_to_hide,c="red",linewidth=1,zorder=10)
plt.show()
del fig,ax

```


![png](/assets/data_pipeline_files/data_pipeline_21_0.png)



![png](/assets/data_pipeline_files/data_pipeline_21_1.png)



![png](/assets/data_pipeline_files/data_pipeline_21_2.png)



![png](/assets/data_pipeline_files/data_pipeline_21_3.png)



![png](/assets/data_pipeline_files/data_pipeline_21_4.png)



![png](/assets/data_pipeline_files/data_pipeline_21_5.png)



![png](/assets/data_pipeline_files/data_pipeline_21_6.png)



![png](/assets/data_pipeline_files/data_pipeline_21_7.png)



![png](/assets/data_pipeline_files/data_pipeline_21_8.png)



![png](/assets/data_pipeline_files/data_pipeline_21_9.png)


# Scatter


```python
# Alpha and beta projections

t_flat = np.cumsum(~np.isnan(predicted[:,:,0]),axis=1)[~np.isnan(predicted[:,:,0])].flatten()
alpha_flat = predicted[:,:,0][~np.isnan(predicted[:,:,0])].flatten()
beta_flat  = predicted[:,:,1][~np.isnan(predicted[:,:,0])].flatten()

## log-alpha typically makes more sense.
# alpha_flat = np.log(alpha_flat)

from matplotlib.colors import LogNorm
counts, xedges, yedges, _ = plt.hist2d(alpha_flat, beta_flat, bins=50,norm=LogNorm())
plt.title('Predicted params : density')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$\beta$')
plt.show()

plt.scatter(alpha_flat,beta_flat,s=0.1)
plt.title('Predicted params')
plt.xlim([alpha_flat.min(),alpha_flat.max()])
plt.ylim([beta_flat.min(),beta_flat.max()])
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$\beta$')
plt.show()

var = t_flat
plt.scatter(alpha_flat,
            beta_flat,
            s=1.0,c=var.flatten(),lw = 0)
plt.title('Predicted params : colored by t_elapsed')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$\beta$')
plt.xlim([alpha_flat.min(),alpha_flat.max()])
plt.ylim([beta_flat.min(),beta_flat.max()])
plt.show()


```


![png](/assets/data_pipeline_files/data_pipeline_23_0.png)



![png](/assets/data_pipeline_files/data_pipeline_23_1.png)



![png](/assets/data_pipeline_files/data_pipeline_23_2.png)



```python
# t_elapsed, "time since start"
t = np.cumsum(~np.isnan(predicted[:,:,0]),axis=1)+predicted[:,:,0]*0

# dt or "wallclock time"
# t = t+np.expand_dims(np.isnan(predicted[:,:,0]).sum(1),-1)

predicted_alpha = predicted[:,:,0]
predicted_beta =  predicted[:,:,1]+predicted[:,:,0]*0

## log-alpha typically makes more sense.
# predicted_alpha = np.log(predicted_alpha)

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

scale_fig = 1.

fig=plt.figure(figsize=(scale_fig*15,scale_fig*5))

# Set the bottom and top outside the actual figure limits, 
# to stretch the 3D axis

ax = fig.add_subplot(111, projection='3d')


# Change the viewing angle to an agreeable one
#ax.view_init(8,320)
#ax.view_init(20,2)
ax.view_init(10,30)


ax = fig.gca(projection='3d')

#for seq in xrange(predicted_alpha.shape[0]):
for seq in [0]:
    plt.cla()
    # highlight path for one sequence
    ax.plot(
           t[seq,:],
           predicted_alpha[seq,:],
           predicted_beta[seq,:],zorder=1
       )
    # plot all others
    ax.scatter(
           t[:,:],
           predicted_alpha[:,:],
           predicted_beta[:,:],
           c='black',lw=0,s=0.5,zorder=0
       )
    # highlight points for one sequence
    ax.scatter(
           t[seq,:],
           predicted_alpha[seq,:],
           predicted_beta[seq,:],
           c='red',lw=0,s=50, marker ='.',zorder=2
       )
    
    ax.set_xlim3d([0,np.nanmax(t)])
    ax.set_ylim3d([0,np.nanmax(predicted_alpha).astype(int)+1])
    
    ax.set_xlabel(r'$t$',fontsize=15)
    ax.set_ylabel(r'$\alpha$',fontsize=15)
    ax.set_zlabel(r'$\beta$',fontsize=15)
    ax.set_title('sequence '+str(seq),fontsize=12)
    
    #ax.invert_xaxis()
    ax.locator_params(axis='both',nbins=4)
    display.display(plt.gcf())
    display.clear_output(wait=True)
    fig.tight_layout()

```


![png](/assets/data_pipeline_files/data_pipeline_24_0.png)


# Calibration

For the uncensored points we assume F(Y) to be uniform. Deviations means we have problem of calibration (which we have obviously below.)


```python
n_bins = 20
cmf = weibull.cmf(t=y[:,:,0],a=predicted[:,:,0],b=predicted[:,:,1])

cmf = cmf[(~np.isnan(y[:,:,1]))*(y[:,:,1]==1) ]

plt.hist(cmf.flatten(),n_bins,weights = np.ones_like(cmf.flatten())/float(len(cmf.flatten())))
plt.xlabel(r'predicted $F(Y)$')
plt.title('histogram ')
plt.axhline(1.0/n_bins,lw=2,c='red',label='expected')
plt.locator_params(axis='both',nbins=5)
plt.legend()
plt.show()

```


![png](/assets/data_pipeline_files/data_pipeline_26_0.png)

