# PyTorch Templates
PyTorch templates for different types of ML models

### Updates
[No release yet]

### *Developer Status*
* Continue the work on the Tabular-Model for ANN regression

#### (*Development Notes*)
* **Required**: None

### Description
This repo aims to provide templates for pytorch modules of different neural network types. The targeted modules include: Tabular, CNN,& RNN modules supporting both classification and regression problems.

Pre-processing to a training-ready format is provided for the data formats specified in the table below.

#### Available Models
| **Model** | **Description** | **Status** |
| - | - | - |
| Tabular | Handles tabular data of 1-d input with a simple ANN. The data should be provided as a pandas dataframe with continuous features, categorical features, or both.  | *In development* |
| CNN | [NA] | *Pending* |
| RNN | [NA] | *Pending* |

---

## Tabular Model

### *Developer Notes*
* DONE
    - NewYork City Taxi Fare dataset (nyc dataset)
    - General preprocessing function (@pd_to_torch)
    - custom preprocessing function (@preprocess_nyc)

* Pending  
    - Tensors to DataLoader function
    - TabularModel class constructor
    - `TabularModel.fit()`: Enclose all pre-processing stages and prepare data for training. Should handle validation split.
    - `TabularModel.training()`: Train on fit data with a given configuration
    - `TabularModel.__call__()`: Predict data output.


* Backlog  
    - `TabularModel.config`: Training, validation, and inference Settings



### Features
[NA]

### Example
[NA]


