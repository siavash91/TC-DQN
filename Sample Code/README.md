## Reproducibility Guide ##

<div align="justify"> For the implementation of this work, we have used python 3.7.4, PyTorch 1.2, OpenAI gym, and SUMO 1.3.1.
See the full-version manuscript to find the settings and hyperparameters. </div> <br>

<div align="justify"> The traffic vehicle generation rate is provided in the table below. The traffic level parameters are the probability of vehicle generation on corresponding lanes and due to different traffic levels as defined in SUMO. To find these probabilities, we use data from OpenData and also conducted on-site in-person observations at intersections in San Francisco Bay Area. This was an effort to make the input to the simulations more realistic as \ac{RL} trial-and-error is high-risk and impractical on real urban traffic. </div> <br>

## The test environments <br>

<img src=envs.png width="950" height="200" /> <br> <br>


## Traffic vehicle generation rate <br>

<!--- <img src=table.PNG width="450" height="250" /> --->

<p float="left">
  &emsp;
  <img src=table.PNG width="450" height="250" />
  &emsp; &emsp;
  <img src=final_plot.png width="420" height="260" />
</p> <br>
