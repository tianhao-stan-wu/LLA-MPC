# Look-Back and Look-Ahead Adaptive Model Predictive Control (LLA-MPC)

<div align="center">
<img src="results/LLA-MPC.jpg" width="600px"/>
</div>


This repository contains the code for the paper "LLA-MPC: Fast Adaptive Control for Autonomous Racing".

# Summary

## Video Presentation (Coming soon!)


## Numerical Simulations

<div align="center" style="display: flex; justify-content: space-around; width: 100%;">
  <div style="width: 48%; margin-right: 2%; box-sizing: border-box;"> <!-- Include margin and box sizing for precise control -->
    <img src="results/table1.png" style="width: 100%;">
    <div style="text-align: center; color: white; padding-top: 5px; background-color: rgba(0,0,0,0.7);">
      ETHZ Track
    </div>
  </div>
  <div style="width: 48%; box-sizing: border-box;">
    <img src="results/table2.png" style="width: 100%;">
    <div style="text-align: center; color: white; padding-top: 5px; background-color: rgba(0,0,0,0.7);">
      ETHZMobil Track
    </div>
  </div>
</div>


### Experiment 1


#### ETHZ Track

<div align="center" style="position: relative; display: flex; justify-content: space-around; width: 100%;">
  <div style="position: relative; width: 32%; margin-bottom: 20px;"> <!-- Added margin for spacing -->
    <img src="results/LLA/CASE 2 (GRAD AFTER)/lla1.gif" style="width: 100%;">
    <div style="text-align: center; color: white; padding-top: 5px; background-color: rgba(0,0,0,0.7);"> <!-- Caption below image -->
      LLA-MPC
    </div>
  </div>
  <div style="position: relative; width: 32%; margin-bottom: 20px;">
    <img src="results/APACRace/CASE 2 (GRAD AFTER)/apac1.gif" style="width: 100%;">
    <div style="text-align: center; color: white; padding-top: 5px; background-color: rgba(0,0,0,0.7);">
      APACRace
    </div>
  </div>
  <div style="position: relative; width: 32%; margin-bottom: 20px;">
    <img src="results/GT/CASE 2 (GRAD AFTER)/ora1.gif" style="width: 100%;">
    <div style="text-align: center; color: white; padding-top: 5px; background-color: rgba(0,0,0,0.7);">
      Oracle
    </div>
  </div>
</div>


#### ETHZMobil Track

<div align="center" style="position: relative; display: flex; justify-content: space-around; width: 100%;">
  <div style="position: relative; width: 32%; margin-bottom: 20px;"> <!-- Added margin for spacing -->
    <img src="results/LLA T2/CASE 2 (GRAD AFTER)/lla1_2.gif" style="width: 100%;">
    <div style="text-align: center; color: white; padding-top: 5px; background-color: rgba(0,0,0,0.7);"> <!-- Caption below image -->
      LLA-MPC
    </div>
  </div>
  <div style="position: relative; width: 32%; margin-bottom: 20px;">
    <img src="results/APACRace T2/CASE 2 (GRAD AFTER)/apac1_2.gif" style="width: 100%;">
    <div style="text-align: center; color: white; padding-top: 5px; background-color: rgba(0,0,0,0.7);">
      APACRace
    </div>
  </div>
  <div style="position: relative; width: 32%; margin-bottom: 20px;">
    <img src="results/GT T2/CASE 2 (GRAD AFTER)/ora1_2.gif" style="width: 100%;">
    <div style="text-align: center; color: white; padding-top: 5px; background-color: rgba(0,0,0,0.7);">
      Oracle
    </div>
  </div>
</div>

### Experiment 2

#### ETHZ Track

<div align="center" style="position: relative; display: flex; justify-content: space-around; width: 100%;">
  <div style="position: relative; width: 32%; margin-bottom: 20px;"> <!-- Added margin for spacing -->
    <img src="results/LLA/CASE 4 (SUDD AFTER) - 22/lla2.gif" style="width: 100%;">
    <div style="text-align: center; color: white; padding-top: 5px; background-color: rgba(0,0,0,0.7);"> <!-- Caption below image -->
      LLA-MPC
    </div>
  </div>
  <div style="position: relative; width: 32%; margin-bottom: 20px;">
    <img src="results/APACRace/CASE 4 (SUDD AFTER) - 22/apac2.gif" style="width: 100%;">
    <div style="text-align: center; color: white; padding-top: 5px; background-color: rgba(0,0,0,0.7);">
      APACRace
    </div>
  </div>
  <div style="position: relative; width: 32%; margin-bottom: 20px;">
    <img src="results/GT/CASE 4 (SUDD AFTER) - 22/ora2.gif" style="width: 100%;">
    <div style="text-align: center; color: white; padding-top: 5px; background-color: rgba(0,0,0,0.7);">
      Oracle
    </div>
  </div>
</div>


#### ETHZMobil Track

<div align="center" style="position: relative; display: flex; justify-content: space-around; width: 100%;">
  <div style="position: relative; width: 32%; margin-bottom: 20px;"> <!-- Added margin for spacing -->
    <img src="results/LLA T2/CASE 4 (SUDD AFTER) - 22/lla2_2.gif" style="width: 100%;">
    <div style="text-align: center; color: white; padding-top: 5px; background-color: rgba(0,0,0,0.7);"> <!-- Caption below image -->
      LLA-MPC
    </div>
  </div>
  <div style="position: relative; width: 32%; margin-bottom: 20px;">
    <img src="results/APACRace T2/CASE 4 (SUDD AFTER) - 22/apac2_2.gif" style="width: 100%;">
    <div style="text-align: center; color: white; padding-top: 5px; background-color: rgba(0,0,0,0.7);">
      APACRace
    </div>
  </div>
  <div style="position: relative; width: 32%; margin-bottom: 20px;">
    <img src="results/GT T2/CASE 4 (SUDD AFTER) - 22/ora2_2.gif" style="width: 100%;">
    <div style="text-align: center; color: white; padding-top: 5px; background-color: rgba(0,0,0,0.7);">
      Oracle
    </div>
  </div>
</div>


### Experiment 3

#### ETHZ Track

<div align="center" style="position: relative; display: flex; justify-content: space-around; width: 100%;">
  <div style="position: relative; width: 32%; margin-bottom: 20px;"> <!-- Added margin for spacing -->
    <img src="results/LLA/CASE 3 (SUDD BEG) - 22/lla3.gif" style="width: 100%;">
    <div style="text-align: center; color: white; padding-top: 5px; background-color: rgba(0,0,0,0.7);"> <!-- Caption below image -->
      LLA-MPC
    </div>
  </div>
  <div style="position: relative; width: 32%; margin-bottom: 20px;">
    <img src="results/APACRace/CASE 3 (SUDD BEG) - 22/apac3.gif" style="width: 100%;">
    <div style="text-align: center; color: white; padding-top: 5px; background-color: rgba(0,0,0,0.7);">
      APACRace
    </div>
  </div>
  <div style="position: relative; width: 32%; margin-bottom: 20px;">
    <img src="results/GT/CASE 3 (SUDD BEG) - 22/ora3.gif" style="width: 100%;">
    <div style="text-align: center; color: white; padding-top: 5px; background-color: rgba(0,0,0,0.7);">
      Oracle
    </div>
  </div>
</div>


#### ETHZMobil Track

<div align="center" style="position: relative; display: flex; justify-content: space-around; width: 100%;">
  <div style="position: relative; width: 32%; margin-bottom: 20px;"> <!-- Added margin for spacing -->
    <img src="results/LLA T2/CASE 3 (SUDD BEG) - 22/lla3_2.gif" style="width: 100%;">
    <div style="text-align: center; color: white; padding-top: 5px; background-color: rgba(0,0,0,0.7);"> <!-- Caption below image -->
      LLA-MPC
    </div>
  </div>
  <div style="position: relative; width: 32%; margin-bottom: 20px;">
    <img src="results/APACRace T2/CASE 3 (SUDD BEG) - 22/apac3_2.gif" style="width: 100%;">
    <div style="text-align: center; color: white; padding-top: 5px; background-color: rgba(0,0,0,0.7);">
      APACRace
    </div>
  </div>
  <div style="position: relative; width: 32%; margin-bottom: 20px;">
    <img src="results/GT T2/CASE 3 (SUDD BEG) - 22/ora3_2.gif" style="width: 100%;">
    <div style="text-align: center; color: white; padding-top: 5px; background-color: rgba(0,0,0,0.7);">
      Oracle
    </div>
  </div>
</div>


## CARLA Simulations


# How to run the Code

## LLA-MPC

## APACRace

## Oracle


