# Data Science Capstone

Our capstone project involves learning about graph analysis methods and how they are applied to chip data to optimize them for power usage, performance, and size. The first part of the project is about understanding graph methods and algorithms, and implementing them on a New York City MTA dataset for subway rides. By representing subway stations as nodes in a graph and rides between them as edges, we found insights into the behavior of riders on average. The second part of the project is then applying graph machine learning to chip data, and reproducing the results of another paper's experiments. We developed models to predict congested areas of the chip to fix them before the physical creation of the chips.

# Folder Structure
```
|
└───data
|   |   chips
|   |   └───clean_data
|   └───mta
└───results
|   |   chips
|   └───mta
└───src
|   |   chips
|   └───mta
└───README.md
└───requirments.txt
```

# Prerequisites
The dependencies are listed under requirements.txt and are all purely python based. First create a virtual environment using version 3.8.8 of python. To install the dependencies run

```
pip install -r requirements.txt
```

# MTA Analysis

This project analyzes the average ridership between different subway stations in the New York City MTA to understand where the highest levels of congestion are. The MTA records information about every station in their network, as well as the average ridership between two stations at any given hour and day of the week. We use a graph representation to model the subway network and use properties of graphs to summarize the congestion at and between stations at various times of the week.

## Dataset
First download the two datasets from <br>
(1) https://data.ny.gov/Transportation/MTA-Subway-Stations/39hk-dx4f/about_data <br>
(2) https://data.ny.gov/Transportation/MTA-Subway-Origin-Destination-Ridership-Estimate-2/jsu2-fbtj/about_data <br>

Make sure that the files are called <br>
(1) MTA_Subway_Stations_updated.csv <br>
(2) MTA_Subway_Origin-Destination_Ridership_Estimate__2024_20241008.csv <br>

Then add the csv files to the `data/mta/` folder.

## Running

Navigate to `src/mta/` to find the code for the project. The entire code is written inside the jupyter notebook called `mta.ipynb`. Run each code cell to get the results of the analysis. The results are also in the folder called `results/mta/`.
<br><br><br>

# Chip Profiling

The second project analyzes cells and nets in a given netlist for a chip to determine areas of high congestion. We represent the chip as a hypergraph and use a message passing neural network with virtual nodes to create a model that we believe accurately predicts congestion. Our work is a reimplementation of this paper (https://arxiv.org/abs/2404.00477). We test our models on both the superblue data from the paper and the xbar data from DigiC.

## Dataset

(1) Download the Superblue dataset from [here](https://zenodo.org/records/10795280?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6Ijk5NjM2MzZiLTg0ZmUtNDI2My04OTQ3LTljMjA5ZjA3N2Y1OSIsImRhdGEiOnt9LCJyYW5kb20iOiJlYzFmMGJlZTU3MzE1OWMzOTU2MWZkYTE3MzY5ZjRjOCJ9.WifQFExjW1CAW0ahf3e5Qr0OV9c2cw9_RUbOXUsvRbnKlkApNZwVCL_VPRJvAve0MJDC0DDOSx_RLiTvBimr0w). Extract all the files into a folder called `2023-03-06_data`. Then add this folder to `data/chips/`.

(2) Download the DigiC dataset from [here](https://drive.google.com/file/d/1Scq35gvCQvIMrmthGs7MUhc8c1VZ8ZwN/view). Extract the files into a folder called `NCSU-DigIC-GraphData-2023-07-25` and put it under `data/chips/`.

## Running

Go to `src/chips/` to find the code for the project and run these python files. Choose which model file is needed based on the type of chip and metric to predict over.

Data Exploration <br>
(1) `exploration.ipynb` shows various statistics for chips like layouts and cell distributions. The plots are saved under `results/chips/` <br>

Data Processing <br>
(2) `partition.py` partitions the chip into smaller areas by separating highly connected areas <br>
(3) `create_features.py` creates all the features that will go into the model and saves them under `data/chips/clean_data/` <br>

Model Training <br>
(4) `congestion_superblue.py` trains a model on the superblue chips to classify each cell's congestion <br>
(5) `congestion_xbar.py` trains a model on the xbar chips to classify each cell's congestion <br>
(6) `demand_superblue.py` trains a model on the superblue chips to predict each cell's demand <br>
(7) `demand_xbar.py` trains a model on the xbar chips to predict each cell's demand <br>
(8) `hpwl_superblue.py` trains a model on the superblue chips to predict each net's wire length <br>
(9) `hpwl_xbar.py` trains a model on the xbar chips to predict each net's wire length <br>

Model Testing <br>
(10) `test_superblue.ipynb` shows how to test the model performance on superblue chips <br>
(11) `test_xbar.ipynb` shows how to test model performance on xbar chips <br>