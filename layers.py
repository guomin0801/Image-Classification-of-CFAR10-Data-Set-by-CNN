import numpy as np


def affine_forward(x, w, b):   
    """
    Computes the forward pass for an affine (fully-connected) layer.
    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N   
    examples, where each example x[i] has shape (d_1, ..., d_k). We will    
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and    
    then transform it to an output vector of dimension M.    
    Inputs:    
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)    
    - w: A numpy array of weights, of shape (D, M)    
    - b: A numpy array of biases, of shape (M,)   
    Returns a tuple of:    
    - out: output, of shape (N, M)    
    - cache: (x, w, b)

    全连接层：矩阵变换，获取对应目标相同的行与列
    输入x: 2*32*16*16
    输入x_row: 2*8192
    超参w：8192*100
    输出：矩阵乘法 2*8192 ->8192*100 =>2*100
    """
    out = None
    # Reshape x into rows
    N = x.shape[0]
    x_row = x.reshape(N, -1)         # (N,D)
    # -1表示不知道多少列，指定行，就能算出列 = 2 * 32* 16 * 16/2 = 8192
    out = np.dot(x_row, w) + b       # (N,M)
    # 2*8192 8192*100 =>2 * 100
    cache = (x, w, b)

    return out, cache


def affine_backward(dout, cache):   
    """    
    Computes the backward pass for an affine layer.    
    Inputs:    
    - dout: Upstream derivative, of shape (N, M)    
    - cache: Tuple of: 
    - x: Input data, of shape (N, d_1, ... d_k)    
    - w: Weights, of shape (D, M)    
    Returns a tuple of:   
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)    
    - dw: Gradient with respect to w, of shape (D, M) 
    - db: Gradient with respect to b, of shape (M,)

    反向传播之affine矩阵变换
    根据dout求出dx,dw,db
    由 out = w * x + b =>
    dx = dout * w
    dw = dout * x
    db = dout * 1
    因为dx 与 x，dw 与 w，db 与 b 大小（维度）必须相同
    dx = dout * wT  矩阵乘法
    dw = dxT * dout 矩阵乘法
    db = dout 按列求和
    """    
    x, w, b = cache    
    dx, dw, db = None, None, None   
    dx = np.dot(dout, w.T)                       # (N,D)
    # dx维度必须跟x维度相同
    dx = np.reshape(dx, x.shape)                 # (N,d1,...,d_k)
    # 转换成二维矩阵
    x_row = x.reshape(x.shape[0], -1)            # (N,D)    
    dw = np.dot(x_row.T, dout)                   # (D,M)    
    db = np.sum(dout, axis=0, keepdims=True)     # (1,M)    

    return dx, dw, db


def relu_forward(x):   
    """    
    Computes the forward pass for a layer of rectified linear units (ReLUs).    
    Input:    
    - x: Inputs, of any shape    
    Returns a tuple of:    
    - out: Output, of the same shape as x    
    - cache: x

    激活函数，解决sigmoid梯度消失问题，网络性能比sigmoid更好
    """   
    out = None    
    out = ReLU(x)    
    cache = x    

    return out, cache


def relu_backward(dout, cache):   
    """  
    Computes the backward pass for a layer of rectified linear units (ReLUs).   
    Input:    
    - dout: Upstream derivatives, of any shape    
    - cache: Input x, of same shape as dout    
    Returns:    
    - dx: Gradient with respect to x    
    """    
    dx, x = None, cache    
    dx = dout    
    dx[x <= 0] = 0    

    return dx


def svm_loss(x, y):   
    """    
    Computes the loss and gradient using for multiclass SVM classification.    
    Inputs:    
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class         
         for the ith input.    
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and         
         0 <= y[i] < C   
    Returns a tuple of:    
    - loss: Scalar giving the loss   
    - dx: Gradient of the loss with respect to x

    """    
    N = x.shape[0]   
    correct_class_scores = x[np.arange(N), y]    
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)    
    margins[np.arange(N), y] = 0   
    loss = np.sum(margins) / N   
    num_pos = np.sum(margins > 0, axis=1)    
    dx = np.zeros_like(x)   
    dx[margins > 0] = 1    
    dx[np.arange(N), y] -= num_pos    
    dx /= N    

    return loss, dx


def softmax_loss(x, y):    
    """    
    Computes the loss and gradient for softmax classification.    Inputs:    
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class         
    for the ith input.    
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and         
         0 <= y[i] < C   
    Returns a tuple of:    
    - loss: Scalar giving the loss    
    - dx: Gradient of the loss with respect to x

    softmax_loss 求梯度优点: 求梯度运算简单，方便
    softmax: softmax用于多分类过程中，它将多个神经元的输出，映射到（0,1）区间内，
    可以看成概率来理解，从而来进行多分类。
    Si = exp(i)/[exp(j)求和]
    softmax_loss：损失函数，求梯度dx必须用到损失函数，通过梯度下降更新超参
    Loss = -[Ypred*ln(Sj真实类别位置的概率值)]求和
    梯度dx : 对损失函数求一阶偏导
    如果 j = i =>dx = Sj - 1
    如果 j != i => dx = Sj
    """
    # x - np.max(x, axis=1, keepdims=True) 对数据进行预处理，
    # 防止np.exp(x - np.max(x, axis=1, keepdims=True))得到结果太分散；
    # np.max(x, axis=1, keepdims=True)保证所得结果维度不变；
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))  # axis=1是按照行计算，
    # 计算softmax，准确的说应该是soft，因为还没有选取概率最大值的操作
    probs /= np.sum(probs, axis=1, keepdims=True)
    # 样本图片个数
    N = x.shape[0]
    # 计算图片损失
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    # 复制概率
    dx = probs.copy()
    # 针对 i = j 求梯度
    dx[np.arange(N), y] -= 1
    # 计算每张样本图片梯度
    dx /= N    

    return loss, dx


def ReLU(x):    
    """ReLU non-linearity."""    
    return np.maximum(0, x)


def conv_forward_naive(x, w, b, conv_param):
    """
        A naive implementation of the forward pass for a convolutional layer.

        The input consists of N data points, each with C channels, height H and
        width W. We convolve each input with F different filters, where each filter
        spans all C channels and has height HH and width WW.

        Input:
        - x: Input data of shape (N, C, H, W)
        - w: Filter weights of shape (F, C, HH, WW)
        - b: Biases, of shape (F,)
        - conv_param: A dictionary with the following keys:
          - 'stride': The number of pixels between adjacent receptive fields in the
            horizontal and vertical directions.
          - 'pad': The number of pixels that will be used to zero-pad the input.


        During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
        along the height and width axes of the input. Be careful not to modfiy the original
        input x directly.

        Returns a tuple of:
        - out: Output data, of shape (N, F, H', W') where H' and W' are given by
          H' = 1 + (H + 2 * pad - HH) / stride
          W' = 1 + (W + 2 * pad - WW) / stride
        - cache: (x, w, b, conv_param)

        功能：获取图片特征
        前向卷积：每次用一个3维的卷积核与图片RGB各个通道分别卷积（卷积核1与R进行点积，卷积核2与G点积，卷积核3与B点积）,
        然后将3个结果求和（也就是 w*x ）,再加上 b，就是新结果某一位置输出，这是卷积核在图片某一固定小范围内（卷积核大小）的卷积，
        要想获得整个图片的卷积结果，需要在图片上滑动卷积核（先右后下），直至遍历整个图片。
        x: 2*3*32*32  每次选取2张图片，图片大小32*32，彩色(3通道)
        w: 32*3*7*7   卷积核每个大小是7*7；对应输入x的3通道，所以是3维，有32个卷积核
        pad = 3(图片边缘行列补0)，stride = 1(卷积核移动步长)
        输出宽*高结果：(32-7+2*3)/1 + 1 = 32
        输出大小：2*32*32*32
        """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    stride, pad = conv_param['stride'], conv_param['pad']
    N, C, H, W = x.shape  # N个样本，C个通道，H的高度，W的宽度
    F, C, HH, WW = w.shape  # F个filter，C个通道，HH的filter高度，WW的filter宽度
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    # 计算卷积结果矩阵的大小并分配全零值占位
    s = stride
    '''// : 求整型'''
    H_new = 1 + (H + 2 * pad - HH) // s
    W_new = 1 + (H + 2 * pad - WW) // s
    out = np.zeros((N, F, H_new, W_new))

    # 卷积开始
    for i in range(N):  # ith image
        for f in range(F):  # fth filter
            for j in range(H_new):
                for k in range(W_new):
                    # print(x_padded[i, :, j*s:HH+j*s, k*s:WW+k*s].shape)
                    # print(w[f].shape)
                    # print(b.shape)
                    # print(np.sum(x_padded[i, :, j*s:HH+j*s, k*s:WW+k*s] * w[f] + b[f]))
                    out[i, f, j, k] = np.sum(x_padded[i, :, j*s:HH+j*s, k*s:WW+k*s] * w[f] + b[f])
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
        A naive implementation of the backward pass for a convolutional layer.

        Inputs:
        - dout: Upstream derivatives.
        - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

        Returns a tuple of:
        - dx: Gradient with respect to x
        - dw: Gradient with respect to w
        - db: Gradient with respect to b

        反向传播之卷积：卷积核3*7*7
        输入dout:2*32*32*32
        输出dx:2*3*32*32
        """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # 数据准备
    # print('1111')
    x, w, b, conv_param = cache
    # 边界补0
    pad = conv_param['pad']
    # 步长
    stride = conv_param['stride']
    F, C, HH, WW = w.shape
    N, C, H, W = x.shape
    H_new = 1 + (H + 2 * pad - HH) // stride
    W_new = 1 + (W + 2 * pad - WW) // stride

    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    s = stride
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
    dx_padded = np.pad(dx, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
    for i in range(N):  # ith image 图片个数
        for f in range(F):  # fth filter 卷积核滤波个数
            for j in range(H_new):
                for k in range(W_new):
                    window = x_padded[i, :, j*s:HH+j*s, k*s:WW+k*s]
                    db[f] += dout[i, f, j, k]
                    dw[f] += window * dout[i, f, j, k]
                    dx_padded[i, :, j*s:HH+j*s, k*s:WW+k*s] += w[f] * dout[i, f, j, k]
    # Unpad
    dx = dx_padded[:, :, pad: pad + H, pad: pad + W]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
        A naive implementation of the forward pass for a max-pooling layer.

        Inputs:
        - x: Input data, of shape (N, C, H, W)
        - pool_param: dictionary with the following keys:
          - 'pool_height': The height of each pooling region
          - 'pool_width': The width of each pooling region
          - 'stride': The distance between adjacent pooling regions

        No padding is necessary here. Output size is given by

        Returns a tuple of:
        - out: Output data, of shape (N, C, H', W') where H' and W' are given by
          H' = 1 + (H - pool_height) / stride
          W' = 1 + (W - pool_width) / stride
        - cache: (x, pool_param)

        功能：减少特征尺寸大小
        前向最大池化：在特征矩阵中选取指定大小窗口，获取窗口内元素最大值作为输出窗口映射值，
        先有后下遍历，直至获取整个特征矩阵对应的新映射特征矩阵。
        输入x：2*32*32*32
        池化参数：窗口：2*2，步长：2
        输出窗口宽，高：(32-2)/2 + 1 = 16
        输出大小：2*32*16*16
        """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    N, C, H, W = x.shape
    HH, WW = pool_param['pool_height'], pool_param['pool_width']
    s = pool_param['stride']
    # H_new = 1 + (H - HH) // s
    H_new = 1 + int((H - HH) / s)
    # W_new = 1 + (W - WW) // s
    W_new = 1 + int((W - WW) / s)

    out = np.zeros((N, C, H_new, W_new))
    for i in range(N):
        for j in range(C):
            for k in range(H_new):
                for l in range(W_new):
                    window = x[i, j, k*s: HH+k*s, l*s: WW+l*s]
                    out[i, j, k, l] = np.max(window)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
        A naive implementation of the backward pass for a max-pooling layer.

        Inputs:
        - dout: Upstream derivatives
        - cache: A tuple of (x, pool_param) as in the forward pass.

        Returns:
        - dx: Gradient with respect to x

        反向传播之池化：增大特征尺寸大小
        在缓存中取出前向池化时输入特征，选取某一范围矩阵窗口，
        找出最大值所在的位置，根据这个位置将dout值映射到新的矩阵对应位置上，
        而新矩阵其他位置都初始化为0.
        输入dout:2*32*16*16
        输出dx:2*32*32*32
        """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    x, pool_param = cache
    HH, WW = pool_param['pool_height'], pool_param['pool_width']
    s = pool_param['stride']
    N, C, H, W = x.shape
    H_new = 1 + (H - HH) // s  # 池化结果矩阵高度和宽度
    W_new = 1 + (W - WW) // s
    dx = np.zeros_like(x)
    for i in range(N):
        for j in range(C):
            for k in range(H_new):
                for l in range(W_new):
                    # 取前向传播时输入的某一池化窗口
                    window = x[i, j, k*s: HH+k*s, l*s: WW+l*s]
                    # 计算窗口最大值
                    m = np.max(window)
                    # 根据最大值所在位置以及dout对应值=>新矩阵窗口数值
                    # [false,false
                    #  true, false]  * 1 => [0,0
                    #                        1,0]
                    dx[i, j, k*s: HH+k*s, l*s: WW+l*s] = (window == m) * dout[i, j, k, l]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


# ************************
# 以上为原始方法naive， 以下方法为加速方法fast


def conv_forward_fast(x, w, b, conv_param):
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']

    # Check dimensions
    # assert (W + 2 * pad - WW) % stride == 0, 'width does not work'
    # assert (H + 2 * pad - HH) % stride == 0, 'height does not work'

    # Pad the input
    p = pad
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), 'constant')

    # Figure out output dimensions
    H += 2 * pad
    W += 2 * pad
    out_h = (H - HH) // stride + 1
    out_w = (W - WW) // stride + 1

    # Perform an im2col operation by picking clever strides
    shape = (C, HH, WW, N, out_h, out_w)
    strides = (H * W, W, 1, C * H * W, stride * W, stride)
    strides = x.itemsize * np.array(strides)
    x_stride = np.lib.stride_tricks.as_strided(x_padded, shape=shape, strides=strides)
    x_cols = np.ascontiguousarray(x_stride)
    x_cols.shape = (C * HH * WW, N * out_h * out_w)

    # Now all our convolutions are a big matrix multiply
    res = w.reshape(F, -1).dot(x_cols) + b.reshape(-1, 1)

    # Reshape the output
    res.shape = (F, N, out_h, out_w)
    out = res.transpose(1, 0, 2, 3)

    # Be nice and return a contiguous array
    # The old version of conv_forward_fast doesn't do this, so for a fair
    # comparison we won't either
    out = np.ascontiguousarray(out)

    cache = (x, w, b, conv_param, x_cols)
    return out, cache


def conv_backward_fast(dout, cache):
    x, w, b, conv_param, x_cols = cache
    stride, pad = conv_param['stride'], conv_param['pad']

    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    _, _, out_h, out_w = dout.shape

    db = np.sum(dout, axis=(0, 2, 3))

    dout_reshaped = dout.transpose(1, 0, 2, 3).reshape(F, -1)
    dw = dout_reshaped.dot(x_cols.T).reshape(w.shape)

    dx_cols = w.reshape(F, -1).T.dot(dout_reshaped)
    dx_cols.shape = (C, HH, WW, N, out_h, out_w)
    # dx = col2im_6d_cython(dx_cols, N, C, H, W, HH, WW, pad, stride)
    dx = col2im(dx_cols, (N, C, H, W), HH, WW, stride=stride, pad=0)

    return dx, dw, db


def max_pool_forward_fast(x, pool_param):
    """
    A fast implementation of the forward pass for a max pooling layer.

    This chooses between the reshape method and the im2col method. If the pooling
    regions are square and tile the input image, then we can use the reshape
    method which is very fast. Otherwise we fall back on the im2col method, which
    is not much faster than the naive method.
    """
    N, C, H, W = x.shape
    pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
    stride = pool_param['stride']

    same_size = pool_height == pool_width == stride
    tiles = H % pool_height == 0 and W % pool_width == 0
    if same_size and tiles:
        out, reshape_cache = max_pool_forward_reshape(x, pool_param)
        cache = ('reshape', reshape_cache)
    else:
        out, im2col_cache = max_pool_forward_im2col(x, pool_param)
        cache = ('im2col', im2col_cache)
    return out, cache


def max_pool_backward_fast(dout, cache):
    """
    A fast implementation of the backward pass for a max pooling layer.

    This switches between the reshape method an the im2col method depending on
    which method was used to generate the cache.
    """
    method, real_cache = cache
    if method == 'reshape':
        return max_pool_backward_reshape(dout, real_cache)
    elif method == 'im2col':
        return max_pool_backward_im2col(dout, real_cache)
    else:
        raise ValueError('Unrecognized method "%s"' % method)


def max_pool_forward_reshape(x, pool_param):
    """
    A fast implementation of the forward pass for the max pooling layer that uses
    some clever reshaping.

    This can only be used for square pooling regions that tile the input.
    """
    N, C, H, W = x.shape
    pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
    stride = pool_param['stride']
    assert pool_height == pool_width == stride, 'Invalid pool params'
    assert H % pool_height == 0
    assert W % pool_height == 0
    x_reshaped = x.reshape(N, C, H // pool_height, pool_height, W // pool_width, pool_width)
    out = x_reshaped.max(axis=3).max(axis=4)

    cache = (x, x_reshaped, out)
    return out, cache


def max_pool_backward_reshape(dout, cache):
    """
    A fast implementation of the backward pass for the max pooling layer that
    uses some clever broadcasting and reshaping.

    This can only be used if the forward pass was computed using
    max_pool_forward_reshape.

    NOTE: If there are multiple argmaxes, this method will assign gradient to
    ALL argmax elements of the input rather than picking one. In this case the
    gradient will actually be incorrect. However this is unlikely to occur in
    practice, so it shouldn't matter much. One possible solution is to split the
    upstream gradient equally among all argmax elements; this should result in a
    valid subgradient. You can make this happen by uncommenting the line below;
    however this results in a significant performance penalty (about 40% slower)
    and is unlikely to matter in practice so we don't do it.
    """
    x, x_reshaped, out = cache

    dx_reshaped = np.zeros_like(x_reshaped)
    out_newaxis = out[:, :, :, np.newaxis, :, np.newaxis]
    mask = (x_reshaped == out_newaxis)
    dout_newaxis = dout[:, :, :, np.newaxis, :, np.newaxis]
    dout_broadcast, _ = np.broadcast_arrays(dout_newaxis, dx_reshaped)
    dx_reshaped[mask] = dout_broadcast[mask]
    dx_reshaped /= np.sum(mask, axis=(3, 5), keepdims=True)
    dx = dx_reshaped.reshape(x.shape)

    return dx


# ******************************************
# python实现im2col和col2im


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    Parameters
    ----------
    input_data : 由(数据量, 通道, 高, 长)的4维数组构成的输入数据
    filter_h : 卷积核的高
    filter_w : 卷积核的长
    stride : 步幅
    pad : 填充

    Returns
    -------
    col : 2维数组
    """
    # 输入数据的形状
    # N：批数目，C：通道数，H：输入数据高，W：输入数据长
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1  # 输出数据的高
    out_w = (W + 2 * pad - filter_w) // stride + 1  # 输出数据的长
    # 填充 H,W
    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    # (N, C, filter_h, filter_w, out_h, out_w)的0矩阵
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
    # 按(0, 4, 5, 1, 2, 3)顺序，交换col的列，然后改变形状
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]
# ******************************************


def max_pool_forward_im2col(x, pool_param):
    """
    An implementation of the forward pass for max pooling based on im2col.

    This isn't much faster than the naive version, so it should be avoided if
    possible.
    """
    N, C, H, W = x.shape
    pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
    stride = pool_param['stride']

    assert (H - pool_height) % stride == 0, 'Invalid height'
    assert (W - pool_width) % stride == 0, 'Invalid width'

    out_height = (H - pool_height) // stride + 1
    out_width = (W - pool_width) // stride + 1

    x_split = x.reshape(N * C, 1, H, W)
    # x_cols = im2col(x_split, pool_height, pool_width, padding=0, stride=stride)
    x_cols = im2col(x_split, pool_height, pool_width, stride=stride, pad=0)
    x_cols_argmax = np.argmax(x_cols, axis=0)
    x_cols_max = x_cols[x_cols_argmax, np.arange(x_cols.shape[1])]
    out = x_cols_max.reshape(out_height, out_width, N, C).transpose(2, 3, 0, 1)

    cache = (x, x_cols, x_cols_argmax, pool_param)
    return out, cache


def max_pool_backward_im2col(dout, cache):
    """
    An implementation of the backward pass for max pooling based on im2col.

    This isn't much faster than the naive version, so it should be avoided if
    possible.
    """
    x, x_cols, x_cols_argmax, pool_param = cache
    N, C, H, W = x.shape
    pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
    stride = pool_param['stride']

    dout_reshaped = dout.transpose(2, 3, 0, 1).flatten()
    dx_cols = np.zeros_like(x_cols)
    dx_cols[x_cols_argmax, np.arange(dx_cols.shape[1])] = dout_reshaped
    # dx = col2im_indices(dx_cols, (N * C, 1, H, W), pool_height, pool_width, padding=0, stride=stride)
    dx = col2im(dx_cols, (N * C, 1, H, W), pool_height, pool_width, stride=stride, pad=0)
    dx = dx.reshape(x.shape)

    return dx


