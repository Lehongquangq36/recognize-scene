import h5py
# how to create file hdf5:
#f = h5py.File('mytest.hdf5', 'a')

# tích chập 1 chiều
def conv1d(a, w, b = 0, stride = 1, pad = 0):
    """
    compute 1d convolutional (with bias)
    """
    w_old = a.shape[0]
    f = w.shape[0]
    a_pad = np.pad(a, pad_width=pad, mode = 'constant', constant_values = 0)
    w_new = int((w_old - f + 2*pad)/stride) + 1 
    a_res = np.zeros((w_new))
    for i in range(w_new):
        start = i*stride
        end = start + f
        a_res[i] = np.sum(a_pad[start:end]*w) + b 
    return a_res 
# tích chập 2 chiều tín hiệu đơn kênh
def conv2d_11(A, W, b = 0, stride = 1, pad = 0):
    """
    A: input, A.shape = (n_H_old, n_W_old)
    W: filter, W.shape = (f, f)
    """
    n_H_old, n_W_old = A.shape
    f, f = W.shape
    A_pad = np.pad(A, pad_width=pad, mode = 'constant', constant_values = 0)
    # new shape 
    n_H_new = int((n_H_old - f + 2*pad)/stride) + 1 
    n_W_new = int((n_W_old - f + 2*pad)/stride) + 1 
    # result
    A_res = np.zeros((n_H_new, n_W_new))
    # compute 
    for h in range(n_H_new):
        for v in range(n_W_new):
            h_start = h*stride 
            h_end = h_start + f
            v_start = v*stride 
            v_end = v_start + f
            A_res[h, v] = np.sum(A_pad[h_start:h_end, v_start:v_end] * W) + b 
    return A_res 
# tích chập 2 chiều đa kênh
def conv2d(A, W, b, stride=1, pad=0):
    """
    A: input, A.shape = (m, in_height, in_width, in_channel)
    W: filters, W.shape = (f, f, in_channel, out_channel)
    b: biases, b.shape = (out_channel)
    """
    # nếu ko cùng kênh inpu_channel thì in ra thông báo.
    assert A.shape[3] == W.shape[2],\
        'number of input channels ({}) in A != number of input channels ({}) in W'.format(
            A.shape[3], W.shape[2]
        )
    m, in_height, in_width, _ = A.shape
    f, _ , in_channel, out_channel = W.shape
    A_pad = np.pad(A, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=0)
    # new shape
    out_height = int((in_height - f + 2*pad)/stride) + 1
    out_width = int((in_width - f + 2*pad)/stride) + 1
    A_res = np.zeros((m, out_height, out_width, out_channel))

    for i in range(m):
        for h in range(out_height):
            for w in range(out_width):
                for c in range(out_channel):
                    h_start = h*stride
                    h_end = h_start + f
                    w_start = w*stride
                    w_end = w_start + f
                    # Lấy 1 ô trong A_pad để tính tích chập sau này
                    a_slide = A_pad[i, h_start: h_end, w_start:w_end, :]
                    # tính tích chập cho 1 phần tử mới của ảnh.
                    A_res[i, h, w, c] = np.sum(a_slide * W[:, :, :, c]) + b[c]
    return A_res
