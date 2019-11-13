import numpy as np
import csv



if __name__ == '__main__':
    data_dir = './'
    metric_data_name = 'metrics.csv'
    weight_data_name = 'weights.csv'
    data_parsed_name = 'data_parsed.txt'


    # Process metrics
    metric_data_file = open(data_dir + metric_data_name, 'r')
    metric_data = list(csv.reader(metric_data_file, delimiter=' '))

    new_metric_data = []

    RMSE_val_string = "\t&\t".join(metric_data[0])
    RMSE_val_string = "\t$RMSE$ val\t&\t" + RMSE_val_string + "\t\\\\\n"
    new_metric_data.append(RMSE_val_string)
    RMSE_train_string = "\t&\t".join(metric_data[1])
    RMSE_train_string = "\t$RMSE$ train\t&\t" + RMSE_train_string + "\t\\\\\n"
    new_metric_data.append(RMSE_train_string)
    
    R2_val_string = "\t&\t".join(metric_data[2])
    R2_val_string = "\t$R^2$ val\t&\t" + R2_val_string  + "\t\\\\\n"
    new_metric_data.append(R2_val_string)
    R2_train_string = "\t&\t".join(metric_data[3])
    R2_train_string = "\t$R^2$ train\t&\t" + R2_train_string  + "\t\\\\\n"
    new_metric_data.append(R2_train_string)


    # Process weights
    weight_data_file = open(data_dir + weight_data_name, 'r')
    weight_data = list(csv.reader(weight_data_file, delimiter=' '))

    new_weight_data = []

    for weight_counter, _ in enumerate(weight_data):
        weight_string = "\t&\t".join(weight_data[weight_counter])
        weight_name = "\t$W{}$\t&\t".format((weight_counter + 1) % len(weight_data))
        weight_string = weight_name + weight_string + "\t\\\\\n"
        new_weight_data.append(weight_string)


    # Concat and write
    new_data = new_metric_data + new_weight_data
    data_parsed = open(data_dir + data_parsed_name, 'w')
    for line in new_data:
        data_parsed.write(line)

    data_parsed.close()
    metric_data_file.close()
    weight_data_file.close()