## Reproducibility Guide ##

<div align="justify"> For the implementation of this work, we have used python 3.7.4, PyTorch 1.2, OpenAI gym, and SUMO 1.3.1. <br> <br>
  
The traffic level parameters are the probability of vehicle generation on corresponding lanes and due to different traffic levels as defined in SUMO. To find these probabilities, we use data from OpenData and also conducted on-site in-person observations at intersections in San Francisco Bay Area. This was an effort to make the input to the simulations more realistic as RL trial-and-error is high-risk and impractical on real urban traffic.  </div>

<img src=../Figures/envs.png width="1100" height="220" /> <br>

<div align="justify"> We train our model on all three environments for two different traffic flows. First, we conduct training on the normal traffic patterns and then increase the flow to make the environment more complex and, hence, the learning more involved. The table on the left shows vehicle generation rate required by SUMO. The plot on the right depicts convergence of the algorithm along with the reward collection and action selection through RL exploration (sampled for normal traffic scenario at case 3). </div> <br>

<!--- ## Traffic vehicle generation rate <br> --->
<!--- <img src=table.PNG width="450" height="250" /> --->

<p float="left">
  &emsp;
  <img src=../Figures/table.PNG width="450" height="280" />
  &emsp; &emsp; &emsp; &emsp;
  <img src=../Figures/final_plot.png width="450" height="280" />
</p> <br>

Please see the full-version manuscript to find more about the problem settings and implementation details.
