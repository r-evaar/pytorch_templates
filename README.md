[Discontinued] Development Moved to https://github.com/r-evaar/torch_engine

# PyTorch Templates
### Updates
*No release yet*

### Description
This repository provides a library for simple and quick use of different types of neural networks, and it can also be used as templates for custom pytorch modules and further development. The targeted modules include: Tabular, CNN, & RNN modules supporting both classification and regression problems.

Pre-processing input data into a training-ready format is automatically included for the data formats specified in the table below.

#### Available Models
| **Model** | **Description** | **Status** |
| - | - | - |
| Tabular | Handles tabular data with 1-dimensional input to a simple ANN. The data should be provided as a pandas dataframe with continuous features, categorical features, or both. | *In development* |
| CNN | [NA] | *Pending* |
| RNN | [NA] | *Pending* |

### *Development Status*
* Continue the work on the Tabular-Model for ANN regression

---

## Tabular Model

### Features
*Development In-Progress*

### Example
*Development In-Progress, Check `example.py` for more details*

### *Developer Notes*
* DONE
    - NewYork City Taxi Fare dataset (nyc dataset)
    - General preprocessing function (@pd_to_torch)
    - custom preprocessing function (@preprocess_nyc)
    - Tensors to DataLoader function
    - TabularModel class constructor
    - `TabularModel.fit()`: Enclose all pre-processing stages and prepare data for training. Should handle validation split.
    - `TabularModel.train_model()`: Train on fit data with a given configuration

* Pending  
    - Add a feature for data validation during training to `TabularModel.train_model()`
    - `TabularModel.test()`: Test the model on the test dataset (if specified during `fit`) and save the results to disk
    - `TabularModel.__call__()`: Predict data output.
    - `TabularModel.configs`: Training, evaluation, and inference Settings

* Backlog  
    - Complete all unit tests
    - Add a training log and checkpoints to `TabularModel.train_model()` that save results to disk 
    - Add suitable getters and setters for TabularModel.configs attributes and make the attributes private
    

