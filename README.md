# TC-DQN+: A Novel Approach to Adaptive Traffic Control with Deep Reinforcement Learning


In this repository, we provide the details of the implementation of the following manuscript: <br> <br>


### Adaptive Traffic Control with Deep RL: Towards State-of-the-art and Beyond

Siavash Alemzadeh, Ramin Moslemi, Ratnesh Sharma, Mehran Mesbahi <br> <br>


---

## Abstract

<div align="justify"> In this work, we study adaptive data-guided traffic planning and control using Reinforcement Learning. We shift from the plain use of classic methods towards state-of-the-art in deep RL community. We embed several recent techniques in our algorithm that improve the original DQN for discrete control and discuss the traffic-related interpretations that follow. We propose a novel DQN-based algorithm for Traffic Control (called TC-DQN+) as a tool for fast and more reliable traffic decision-making. We introduce a new form of reward function which is further discussed using illustrative examples with comparisons to traditional traffic control methods. </div> <br>

<p float="left">
  &emsp;
  <img src="http://depts.washington.edu/uwrainlab/wordpress/wp-content/uploads/2020/07/RLScheme-1.png" width="380" height="220" />
  &emsp; &emsp;
  <img src=Demos/Env2-Sc3-Demo.gif width="380" height="220" />
</p> <br> <br>

<div align="justify"> Case-studies are provided wherein the benefits of our method as well as comparisons with some traditional architectures in ATSC are simulated (from real traffic scenarios). </div>

## TC-DQN+ vs Self-Organizing Traffic Light

&nbsp; &emsp; **TC-DQN+** &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &nbsp; **SOTL**

<p float="left">
  &emsp;
  <img src=Scen1-Env1-TC-DQN.gif width="380" height="220" />
  &emsp; &emsp;
  <img src=Sce1-Env1-SOTL.gif width="380" height="220" />
</p> <br>


## TC-DQN+ vs Fixed-Time Traffic Plan

&nbsp; &emsp; **TC-DQN+** &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &nbsp; **FT**

<p float="left">
  &emsp;
  <img src=Sce3-Env3-TC-DQN.gif width="380" height="220" />
  &emsp; &emsp;
  <img src=Sce3-Env3-FT.gif width="380" height="220" />
</p> <br>
