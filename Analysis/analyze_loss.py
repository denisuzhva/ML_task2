import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'



def prepareRow(row):
    row_mean = np.mean(row)
    row_std = np.std(row)
    row = np.append(row, row_mean)
    row = np.append(row, row_std)
    row = np.round(row, 2)
    return row


def plotLoss(metric_data, epochs, epoch_quant, is_rmse=True, is_train=True):
    epoch_data = np.arange(0, epochs, epochs // epoch_quant)
    plot_data = metric_data[0, :, :, 0 if is_rmse else 1, 0 if is_train else 1]
    plt.figure(figsize=(12, 8))
    for fold_iter in range(plot_data.shape[0]):
        plt.plot(epoch_data, plot_data[fold_iter, :], label='Fold # %d' % (fold_iter))
    
    plt.axis([0, epochs, np.min(plot_data) * 0.95, np.max(plot_data) * 1.2])

    plt.title('%s on epoch (%s loss)' % ('RMSE' if is_rmse else 'R2', 
                                         'train' if is_train else 'validation'))
    plt.legend(fontsize=10)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('%s value' % 'RMSE' if is_rmse else 'R2', fontsize=14)
    
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.2)
    plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)
    
    plt.show()


if __name__ == '__main__':
    metric_data = np.load('../TrainData/netflix/metrics.npy')
    epochs = 500
    epoch_quant = 100
    print(metric_data.shape)
    plotLoss(metric_data, epochs, epoch_quant, True, True)
    
    batch = 0
    rows = [metric_data[batch, :, -1, 0, 1],   # RMSE val 
            metric_data[batch, :, -1, 0, 0],   # RMSE train 
            metric_data[batch, :, -1, 1, 1],   # R2 val
            metric_data[batch, :, -1, 1, 0]   # R2 train
            ]

    for row_count, row  in enumerate(rows, start=0):
        rows[row_count] = prepareRow(row)


    loss_rows = np.stack((rows[0], 
                          rows[1], 
                          rows[2],
                          rows[3]), axis=0)
    RMSE_val_string = ' '.join(map(str, rows[0]))
    RMSE_train_string = ' '.join(map(str, rows[1]))
    R2_val_string = ' '.join(map(str, rows[2]))
    R2_train_string = ' '.join(map(str, rows[3]))


    np.savetxt('./metrics.csv', loss_rows, delimiter=' ', fmt='%.2f')

