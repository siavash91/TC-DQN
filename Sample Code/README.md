## Reproducibility Guide ##

For the implementation of this work, we have used python 3.7.4, PyTorch 1.2, OpenAI gym, and SUMO 1.3.1.
See the full-version manuscript to find the settings and hyperparameters. <br>

<div align="justify"> The traffic vehicle generation rate in SUMO is provided in the table below. The traffic level parameters are the probability of vehicle generation on corresponding lanes and due to different traffic levels as defined in \ac{SUMO}. To find these probabilities, we use data from \cite{openData} and also conducted on-site in-person observations at intersections given in \Cref{fig:caseStudies} in San Francisco Bay Area. This was an effort to make the input to the simulations more realistic as \ac{RL} trial-and-error is high-risk and impractical on real urban traffic. </div> <br> <br>

&emsp; <img src=table.PNG width="500" height="300" />
