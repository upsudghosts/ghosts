# M2 AIC Survival group starting kit

In this project we explore machine learning methods on a survival analysis problem. The goal of survival analysis is to predict the expected duration before death for living being. 
In this starting kit we provide a baseline method on data from the NHANES mortality dataset.

Special thanks to Christine and Alex for providing a clean aggregated version of the dataset.

## Docker usage (thanks to Isabelle Guyon for the instructions)

Our baseline method uses a model from the lifelines library that is not included in the regular codalab-legacy docker image.
Our docker image is available [here](https://hub.docker.com/r/nnour/codalab-legacy-survival/).

To execute our code in the docker:

Download the starting kit zip file (or clone the repository)
Create a new temporary folder :

    mkdir ~/aux
    
Move the starting kit zip inside the aux folder :

    mv ~/Download/projet_AIC_Survival.zip ~/aux

Launch the docker :

    docker run -it -v ~/aux:~/home/aux nnour/codalab-legacy-survival

Once the docker image is downloaded and is running you can unzip the starting kit :

    cd /home/aux
    unzip projet_AIC_Survival.zip
    
And launch the code :

    cd projet_AIC_Survival
    python3 ingestion_program/ingestion.py sample_data sample_result_submission ingestion_program sample_code_submission
    python3 scoring_program/score.py sample_data sample_result_submission scoring_output

You can exit the docker by pressing ctr-D or typing ``` exit ```.

To launch the notebook from the docker :

    docker run -it -p 8888:8888 -v ~/aux:/home/aux nnour/codalab-legacy-survival

Then open a web browser at http://localhost:8888/ and open the README.pynb with python3 kernel. 

