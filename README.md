# Analyzing-Airplane-Logs-and-Visualization

## **INTRODUCTION**

- The cybersecurity team at Boeing has tasked the UW ENGINE team to develop a cybersecurity threat detection, identification, and mitigation algorithm to assist engineers and airline personnel in dealing with cybersecurity threats.
- The UW ENGINE team was also tasked with researching and designing an embedded device to generate log files on aircraft devices to normalize and generate log files for devices that are not utilizing Boeing’s standards.



## **TECHNOLOGY STACK and TOOLS**

- Developed the backend using Flask and leveraged EChartsJS for interactive data visualization on the front-end

- Implemented Firebase as the database solution for handling user authentication.

- Developed a Flask Cache mechanism aimed at enhancing the browsing speed of voluminous log files.

- Utilized Pandas to preprocess raw log files and applied clustering algorithms from sci-kit-learn for log analysis purposes.



## **IMPLEMENTATION**

### Back-End:

- The gaussian Mixture Model is used to group episodes into different groups

- Silhouette score is used to get the best clustering numbers for different datasets.

- The API utilizes Ajax requests to process asynchronous requests.

- The API supports the following HTTP requests: -GET -POST -PUT -DELETE 

- NPM is utilized for the automatic installation of dependencies and for running the backend script upon deployment to the server using the GitHub pipeline.

- Enhanced the clustering algorithm to handle various log file formats and types efficiently.



### Front-End:

- Designed different pages for different datasets (dataload, firewall, staging)

- Implemented interactive charts using JavaScript

- Implemented user login function with user email verification for security

- Optimized backend analysis for efficient website performance
