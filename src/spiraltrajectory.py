import numpy as np 


def spiral_samples_trajectory(width=512, 
                              height=512, 
                              starting_angle=0, 
                              n_rounds=10, 
                              r=np.linspace(0, 1, 1000000)):

    t = np.linspace(0, 1, len(r))

    for curr_angle in range(starting_angle + 1):
        x = np.cos(2 * np.pi * n_rounds * t + curr_angle) * r
        y = np.sin(2 * np.pi * n_rounds * t + curr_angle) * r

    # 0 - 511
    x = (x/2 + 0.5) * (height - 1)
    y = (y/2 + 0.5) * (width - 1)
    
    i = np.round(width - y).astype(int)
    j = np.round(x).astype(int)
    I = np.zeros((width, height))

    for k in range(len(i)):
        try:
            I[i[k],j[k]] = 1
        except Exception as e:
            print(e)

    I = np.fft.ifftshift(I)
    I = np.reshape(I, [width * height, 1])
    samples_rows = np.nonzero(I)[0]
    samples_rows = np.sort(samples_rows)
    I = np.reshape(I, [width, height])

    return samples_rows, i, j, I
