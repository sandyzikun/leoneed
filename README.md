# `Leo/need`

A Simple Trial on Tensor-Graph-based Network...

## Requirements

Ensure [NumPy](https://numpy.org/) ([`NumPy` on GitHub](https://github.com/numpy/numpy/)), and [Matplotlib](https://matplotlib.org/) ([`Matplotlib` on GitHub](https://github.com/matplotlib/matplotlib/)) is installed already before installing `Leo/need`.

One of most simple ways to install them is installing it with `conda`:

```sh
$ conda install numpy matplotlib
```

## Installation

Currently the latest version of `Leo/need` can be installed with `pip` as following:

```sh
$ pip install leoneed --upgrade
```

or [from source](https://github.com/sandyzikun/leoneed/) like other packages.

## Importation

To access `Leo/need` and its functions import it in your Python code like this:

```py
>>> import leoneed as ln
どうだっていい存在じゃない、簡単に愛は終わらないよ。
```

## Components

### `needle`: Nodes in Tensor-Graph

```py Python
>>> mulmat = ln.needle.Mul_Matrix(( 3 , 4 ))
>>> mulmat.tensor
matrix([[ 1.40483957,  0.22112104, -0.14532731,  0.12319917],
        [ 0.60602697,  2.42277001, -1.91660854, -2.42252709],
        [ 0.64629422,  0.20150064, -0.15671318,  0.77204576]])
```

### `stella`: Instances of Loss Functions

It Returns the Value and the Gradient of the Specified Loss Function.

```py Python
>>> simploss = ln.stella.Loss_Simple(3) # Loss Function: (y_pred - y_true)^2 / 2
>>> simploss([ 1, 3, 4 ], [ 5, 7, 1 ])
(matrix([[8. , 8. , 4.5]]), matrix([[-4, -4,  3]]))
```

### `stage`: Models' Containers

```py Python
BATCH_SIZE = 128
NUM_EPOCHES = 39
MINI_BATCH = 5
ae = ln.stage.Auto_Encoder(3, 2)
print("W_Encoder(pre-training):", ae[0].tensor, sep="\n")
print("b_Encoder(pre-training):", ae[1].tensor, sep="\n")
print("W_Decoder(pre-training):", ae[3].tensor, sep="\n")
print("b_Decoder(pre-training):", ae[4].tensor, sep="\n")
randdata = np.random.randn(BATCH_SIZE)
randdata /= np.abs(randdata).max() * 1.28
traindata = np.zeros(( BATCH_SIZE , 3 )) # Constructing a Dataset Filled with Sample Vectors like (+a, 0, -a) Manually.
traindata[ : , 0 ] += randdata
traindata[ : , 2 ] -= randdata
ae_history = []
for idx_epoch in range(NUM_EPOCHES):
    for k in range(BATCH_SIZE):
        ae, ae_loss = ae.fit_sample(traindata[ k : (k + 1) , : ], traindata[ k : (k + 1) , : ])
        ae_history.append(ae_loss)
print("W_Encoder(pst-training):", ae[0].tensor, sep="\n")
print("b_Encoder(pst-training):", ae[1].tensor, sep="\n")
print("W_Decoder(pst-training):", ae[3].tensor, sep="\n")
print("b_Decoder(pst-training):", ae[4].tensor, sep="\n")
with plt.rc_context({}):
    plt.plot(ae_history, label="Loss")
    plt.legend()
    plt.xlabel("Iterations")
    plt.title("Gradient Descend, on the same Batch of %d Samples." % BATCH_SIZE)
    plt.savefig("./gradloss-ae.jpeg")
    plt.show()
```

Output:

```py
W_Encoder(pre-training):
[[ 1.40483957  0.22112104]
 [-0.14532731  0.12319917]
 [ 0.60602697  2.42277001]]
b_Encoder(pre-training):
[[0. 0.]]
W_Decoder(pre-training):
[[-1.91660854 -2.42252709  0.64629422]
 [ 0.20150064 -0.15671318  0.77204576]]
b_Decoder(pre-training):
[[0. 0. 0.]]
W_Encoder(pst-training):
[[ 0.87722083  0.77930038]
 [-0.14532731  0.12319917]
 [ 1.13364571  1.86459068]]
b_Encoder(pst-training):
[[-0.00498058  0.00632017]]
W_Decoder(pst-training):
[[-1.88554672 -2.21171336  0.71931055]
 [-0.56403606  0.55549326  0.85221518]]
b_Decoder(pst-training):
[[ 0.00564453 -0.01603693 -0.01378995]]
```

![](https://github.com/sandyzikun/leoneed/raw/init/gradloss-ae.jpeg)

## Changelog

### Version `0.0.3`

* Finished Implementation of Gradient after Matrix-Multiplication;

* Added API of Generating AE (Auto-Encoder): `.stage.Auto_Encoder(numvisible, numhidden, w1=None, w2=None)`;

### Version `0.0.2`

* Added sub-module `needle` (Nodes of Tensor-Graph), `stella` (Loss Functions), and `stage`;

## References

[^extra-1]: Harry-P (針原 翼), [*Issenkou*](https://zh.moegirl.org.cn/一闪光), 2017, [`av17632876`](https://www.bilibili.com/video/av17632876/);

## Extra

![](https://github.com/sandyzikun/leoneed/raw/init/Issenkou.jpeg)

> 息吹く炎、君の鼓動の中。 \
> Flames Breathing, in your Heartbeat.
> 火炎般的氣息，綻放於你跳動的心臟。 \
> ---- Harry-P in "Issenkou"[^extra-1]
