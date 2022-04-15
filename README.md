# Undergraduate Thesis

_An undergraduate thesis (also called Bachelor's dissertation) is a large academic 
writing piece that requires massive research on the chosen topic._

# Directory Structure

Based on:  
https://drivendata.github.io/cookiecutter-data-science/ <br/>
https://towardsdatascience.com/manage-your-data-science-project-structure-in-early-stage-95f91d4d0600

```text
├── data
│   ├── external                Data from third party sources
│   ├── interim                 Intermediate data that has been transformed
│   ├── processed               The final, canonical data sets for modeling
│   └── raw                     The original, immutable data dump
├── etc                         Stuff that might be useful
├── notebooks                   Jupyter notebooks
│   ├── eda                     
│   ├── poc                     
│   ├── modeling                
│   └── evaluation              
├── output                      Generated graphics and figures to be used in reporting
├── pipelines                   Automation scripts for models retraining
├── references                  Papers, books and all other explanatory materials
├── src                         Source code for use in this project
│   ├── archive                 No longer useful
│   ├── data                    Scripts to download or generate data
│   ├── TBD                       
├── requirements.txt            The requirements file for reproducing the analysis environment
```
