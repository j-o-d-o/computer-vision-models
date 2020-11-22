import matplotlib.pyplot as plt
import json
import argparse


def plot_metric(data, name):
  plt.plot(data["train"])
  plt.plot(data["val"])
  plt.title("Metric: " + name)
  plt.ylabel(name)
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.show()


def read_data_from_json(path):
  with open(path, "r") as file:
    raw_data = json.load(file)
    data = {}
    for metric in raw_data["training"]:
      data[metric] = {"train": [], "val": []}
      # Save val values
      for val in raw_data["validation"]["val_" + metric]:
        data[metric]["val"].append(val["value"])
      # Average out training metrics
      curr_epoch = None
      curr_epoch_avg = None
      number_values = 0
      for val in raw_data["training"][metric]:
        if curr_epoch is None:
          curr_epoch = val["epoch"]
        if val["epoch"] == curr_epoch:
          if curr_epoch_avg is None:
            curr_epoch_avg = val["value"]
          else:
            curr_epoch_avg += val["value"]
          number_values += 1
        else:
          data[metric]["train"].append(curr_epoch_avg / number_values)
          number_values = 0
          curr_epoch_avg = None
          curr_epoch = val["epoch"]
      if curr_epoch_avg is not None:
        data[metric]["train"].append(curr_epoch_avg / number_values)
  return data


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Plot metrics from training")
  parser.add_argument("--path", type=str, help="Path of the metric file e.g. ./trained_models/semseg_16-08-2020-13-16-55/metric.json")
  args = parser.parse_args()

  plot_data = read_data_from_json(args.path)
  for metric in plot_data:
    plot_metric(plot_data[metric], metric)
