import numpy as np
from os import urandom

def WORD_SIZE():
    return(16)

def ALPHA():
    return(7)

def BETA():
    return(2)

MASK_VAL = 2 ** WORD_SIZE() - 1

def rol(x,k):
    return(((x << k) & MASK_VAL) | (x >> (WORD_SIZE() - k)))

def ror(x,k):
    return((x >> k) | ((x << (WORD_SIZE() - k)) & MASK_VAL))

def enc_one_round(p, k):
    c0, c1 = p[0], p[1]
    c0 = ror(c0, ALPHA())
    c0 = (c0 + c1) & MASK_VAL
    c0 = c0 ^ k
    c1 = rol(c1, BETA())
    c1 = c1 ^ c0
    return(c0,c1)

def dec_one_round(c,k):
    c0, c1 = c[0], c[1]
    c1 = c1 ^ c0
    c1 = ror(c1, BETA())
    c0 = c0 ^ k
    c0 = (c0 - c1) & MASK_VAL
    c0 = rol(c0, ALPHA())
    return(c0, c1)

def expand_key(k, t):
    ks = [0 for i in range(t)]
    ks[0] = k[len(k)-1]
    l = list(reversed(k[:len(k)-1]))
    for i in range(t-1):
        l[i%3], ks[i+1] = enc_one_round((l[i%3], ks[i]), i)
    return(ks)

def encrypt(p, ks):
    x, y = p[0], p[1]
    for k in ks:
        x,y = enc_one_round((x,y), k)
    return(x, y)

def decrypt(c, ks):
    x, y = c[0], c[1]
    for k in reversed(ks):
        x, y = dec_one_round((x,y), k)
    return(x,y)

# convert_to_binary takes as input an array of ciphertext pairs
# where the first row of the array contains the lefthand side of the ciphertexts,
# the second row contains the righthand side of the ciphertexts,
# the third row contains the lefthand side of the second ciphertexts,
# and so on
# it returns an array of bit vectors containing the same data
def convert_to_binary(arr, n):
  assert n in [2,4]
  assert len(arr) == n*2
  X = np.zeros((2 * n * WORD_SIZE(),len(arr[0])), dtype=np.uint8)
  for i in range(2 * n * WORD_SIZE()):
    index = i // WORD_SIZE()
    offset = WORD_SIZE() - (i % WORD_SIZE()) - 1
    X[i] = (arr[index] >> offset) & 1
  X = X.transpose()
  return X

# baseline training data generator for 2 plaintexts
def make_train_data_2pt(n, nr, diff=(0x0040,0)):
  Y = np.frombuffer(urandom(n), dtype=np.uint8) 
  Y = Y & 1
  keys = np.frombuffer(urandom(8*n), dtype=np.uint16).reshape(4,-1)
  plain0l = np.frombuffer(urandom(2*n), dtype=np.uint16)
  plain0r = np.frombuffer(urandom(2*n), dtype=np.uint16)
  plain1l = plain0l ^ diff[0] 
  plain1r = plain0r ^ diff[1]
  num_rand_samples = np.sum(Y==0)
  plain1l[Y==0] = np.frombuffer(urandom(2*num_rand_samples), dtype=np.uint16)
  plain1r[Y==0] = np.frombuffer(urandom(2*num_rand_samples), dtype=np.uint16)
  ks = expand_key(keys, nr)
  ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks)
  ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks)
  X = convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r], 2)
  return X,Y

# real differences data generator for 2 plaintexts
def real_differences_data_2pt(n, nr, diff=(0x0040,0)):
  # generate labels
  Y = np.frombuffer(urandom(n), dtype=np.uint8) 
  Y = Y & 1
  
  # generate keys
  keys = np.frombuffer(urandom(8*n), dtype=np.uint16).reshape(4,-1)

  # generate plaintexts
  plain0l = np.frombuffer(urandom(2*n), dtype=np.uint16)
  plain0r = np.frombuffer(urandom(2*n), dtype=np.uint16)

  # apply input difference
  plain1l = plain0l ^ diff[0] 
  plain1r = plain0r ^ diff[1]
  num_rand_samples = np.sum(Y==0)

  # expand keys and encrypt
  ks = expand_key(keys, nr)
  ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks)
  ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks)

  # generate blinding values
  k0 = np.frombuffer(urandom(2*num_rand_samples), dtype=np.uint16)
  k1 = np.frombuffer(urandom(2*num_rand_samples), dtype=np.uint16)

  # apply blinding to the samples labelled as random
  ctdata0l[Y==0] = ctdata0l[Y==0] ^ k0 
  ctdata0r[Y==0] = ctdata0r[Y==0] ^ k1
  ctdata1l[Y==0] = ctdata1l[Y==0] ^ k0 
  ctdata1r[Y==0] = ctdata1r[Y==0] ^ k1

  #convert to input data for neural networks
  X = convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r], 2)

  return X,Y
  
# baseline training data generator for 4 plaintexts
def make_train_data_4pt(n, nr, diffa=(0x0040,0), diffb=(0x0020,0)):
  Y = np.frombuffer(urandom(n), dtype=np.uint8) 
  Y = Y & 1
  keys = np.frombuffer(urandom(8*n), dtype=np.uint16).reshape(4,-1)
  plain0l = np.frombuffer(urandom(2*n),dtype=np.uint16)
  plain0r = np.frombuffer(urandom(2*n),dtype=np.uint16)
  plain1l = plain0l ^ diffa[0] 
  plain1r = plain0r ^ diffa[1]
  plain2l = plain0l ^ diffb[0]
  plain2r = plain0r ^ diffb[1]
  plain3l = plain2l ^ diffa[0] 
  plain3r = plain2r ^ diffa[1]

  num_rand_samples = np.sum(Y==0)

  plain1l[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16)
  plain1r[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16)
  plain2l[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16)
  plain2r[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16)
  plain3l[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16)
  plain3r[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16)

  ks = expand_key(keys, nr)
  ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks)
  ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks)
  ctdata2l, ctdata2r = encrypt((plain2l, plain2r), ks)
  ctdata3l, ctdata3r = encrypt((plain3l, plain3r), ks)
  X = convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r, ctdata2l, ctdata2r, ctdata3l, ctdata3r], 4)
  
  return X,Y

# real differences data generator for 4 plaintexts
def real_differences_data_4pt(n, nr, diffa=(0x0040,0), diffb=(0x0020,0)):
  # generate labels
  Y = np.frombuffer(urandom(n), dtype=np.uint8) 
  Y = Y & 1
  
  # generate keys
  keys = np.frombuffer(urandom(8*n), dtype=np.uint16).reshape(4,-1)

  # generate plaintexts
  plain0l = np.frombuffer(urandom(2*n), dtype=np.uint16)
  plain0r = np.frombuffer(urandom(2*n), dtype=np.uint16)

  # apply input difference
  plain1l = plain0l ^ diffa[0] 
  plain1r = plain0r ^ diffa[1]
  plain2l = plain0l ^ diffb[0]
  plain2r = plain0r ^ diffb[1]
  plain3l = plain2l ^ diffa[0] 
  plain3r = plain2r ^ diffa[1]

  num_rand_samples = np.sum(Y==0)

  # expand keys and encrypt
  ks = expand_key(keys, nr)
  ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks)
  ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks)
  ctdata2l, ctdata2r = encrypt((plain2l, plain2r), ks)
  ctdata3l, ctdata3r = encrypt((plain3l, plain3r), ks)

  # generate blinding values
  k0 = np.frombuffer(urandom(2*num_rand_samples), dtype=np.uint16)
  k1 = np.frombuffer(urandom(2*num_rand_samples), dtype=np.uint16)

  # apply blinding to the samples labelled as random
  ctdata0l[Y==0] = ctdata0l[Y==0] ^ k0
  ctdata0r[Y==0] = ctdata0r[Y==0] ^ k1
  ctdata1l[Y==0] = ctdata1l[Y==0] ^ k0
  ctdata1r[Y==0] = ctdata1r[Y==0] ^ k1
  ctdata2l[Y==0] = ctdata2l[Y==0] ^ k0
  ctdata2r[Y==0] = ctdata2r[Y==0] ^ k1
  ctdata3l[Y==0] = ctdata3l[Y==0] ^ k0
  ctdata3r[Y==0] = ctdata3r[Y==0] ^ k1

  # convert to input data for neural networks
  X = convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r, ctdata2l, ctdata2r, ctdata3l, ctdata3r], 4)

  return X,Y
