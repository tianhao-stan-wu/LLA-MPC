# Look-Back and Look-Ahead Adaptive Model Predictive Control (LLA-MPC)

<div align="center">
<img src="results/LLA-MPC.jpg" width="600px"/>
</div>


This repository contains the code for the paper "LLA-MPC: Fast Adaptive Control for Autonomous Racing".

# Summary

## Video Presentation (Coming soon!)


## Numerical Simulations

<div align="center">
  <table border="1">
    <caption>ETHZ Track</caption>
    <thead>
      <tr>
        <th></th>
        <th>Oracle</th>
        <th>LLA-MPC N=20K</th>
        <th>LLA-MPC N=10K</th>
        <th>APACRace</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Lap 1 Time (s)</td>
        <td style="color:gold;">7.68</td>
        <td><strong>7.76</strong></td>
        <td>8.01</td>
        <td>8.54</td>
      </tr>
      <tr>
        <td>Lap 2 Time (s)</td>
        <td style="color:gold;">7.50</td>
        <td><strong>7.58</strong></td>
        <td>7.85</td>
        <td>7.65</td>
      </tr>
      <tr>
        <td>Lap 3 Time (s)</td>
        <td style="color:gold;">7.76</td>
        <td><strong>7.92</strong></td>
        <td>8.14</td>
        <td>7.96</td>
      </tr>
      <tr>
        <td>Violation Time (s)</td>
        <td style="color:gold;">0.84</td>
        <td><strong>0.78</strong></td>
        <td>1.07</td>
        <td>1.24</td>
      </tr>
      <tr>
        <td>Mean Deviation (m)</td>
        <td style="color:gold;">0.04</td>
        <td><strong>0.04</strong></td>
        <td>0.05</td>
        <td>0.05</td>
      </tr>
      <tr>
        <td>Avg. Comput. Time (s)</td>
        <td style="color:gold;">0.02</td>
        <td><strong>0.03</strong></td>
        <td><strong>0.03</strong></td>
        <td>0.06</td>
      </tr>
    </tbody>
  </table>
  <br/>
  <table border="1">
    <caption>ETHZMobil Track</caption>
    <thead>
      <tr>
        <th></th>
        <th>Oracle</th>
        <th>LLA-MPC N=20K</th>
        <th>LLA-MPC N=10K</th>
        <th>APACRace</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Lap 1 Time (s)</td>
        <td style="color:gold;">5.90</td>
        <td><strong>5.92</strong></td>
        <td>6.88</td>
        <td>6.46</td>
      </tr>
      <tr>
        <td>Lap 2 Time (s)</td>
        <td style="color:gold;">5.76</td>
        <td><strong>5.80</strong></td>
        <td>6.76</td>
        <td>6.18</td>
      </tr>
      <tr>
        <td>Lap 3 Time (s)</td>
        <td style="color:gold;">6.03</td>
        <td><strong>6.04</strong></td>
        <td>6.76</td>
        <td>6.14</td>
      </tr>
      <tr>
        <td>Violation Time (s)</td>
        <td style="color:gold;">0.78</td>
        <td><strong>0.64</strong></td>
        <td>2.42</td>
        <td>1.81</td>
      </tr>
      <tr>
        <td>Mean Deviation (m)</td>
        <td style="color:gold;">0.04</td>
        <td><strong>0.04</strong></td>
        <td>0.08</td>
        <td>0.07</td>
      </tr>
      <tr>
        <td>Avg. Comput. Time (s)</td>
        <td style="color:gold;">0.02</td>
        <td><strong>0.03</strong></td>
        <td><strong>0.03</strong></td>
        <td>0.06</td>
      </tr>
    </tbody>
  </table>
</div>


### Experiment 1

#### ETHZ Track

<div align="center">
  <img src="results/LLA/CASE 2 (GRAD AFTER)/lla1.gif" width="32%"/>
  <img src="results/APACRace/CASE 2 (GRAD AFTER)/apac1.gif" width="32%"/>
  <img src="results/GT/CASE 2 (GRAD AFTER)/ora1.gif" width="32%"/>
</div>

#### ETHZMobil Track

<div align="center">
  <img src="gifs/numerical_simulation_1.gif" width="32%"/>
  <img src="gifs/numerical_simulation_2.gif" width="32%"/>
  <img src="gifs/numerical_simulation_3.gif" width="32%"/>
</div>

### Experiment 2

#### ETHZ Track

<div align="center">
  <img src="gifs/numerical_simulation_1.gif" width="32%"/>
  <img src="gifs/numerical_simulation_2.gif" width="32%"/>
  <img src="gifs/numerical_simulation_3.gif" width="32%"/>
</div>

#### ETHZMobil Track

<div align="center">
  <img src="gifs/numerical_simulation_1.gif" width="32%"/>
  <img src="gifs/numerical_simulation_2.gif" width="32%"/>
  <img src="gifs/numerical_simulation_3.gif" width="32%"/>
</div>


### Experiment 3

#### ETHZ Track

<div align="center">
  <img src="gifs/numerical_simulation_1.gif" width="32%"/>
  <img src="gifs/numerical_simulation_2.gif" width="32%"/>
  <img src="gifs/numerical_simulation_3.gif" width="32%"/>
</div>

#### ETHZMobil Track

<div align="center">
  <img src="gifs/numerical_simulation_1.gif" width="32%"/>
  <img src="gifs/numerical_simulation_2.gif" width="32%"/>
  <img src="gifs/numerical_simulation_3.gif" width="32%"/>
</div>


## CARLA Simulations


# How to run the Code

## LLA-MPC

## APACRace

## Oracle


