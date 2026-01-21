# Improvements?

1.
Steps passed/max steps als input ins netz?

2.
we could track the real energy by torque* w(winkelgeschw.)

3.
So layer stacken für time awarnes, als vergangene werte werden mitgetracked

4.
Epsilon einstellen?

5.
action noise?

6.
Policy evaulieren (test Set)?

7. done
config datei? 

8.
energy über die acceleration berechnen

9.
wirkliche kinetische energy berechnen und als eval metric nehmen

10.
results speichern in json oder so

11.
automatisiertes testen der param with OPTUNA

# Changes we got to make later on for the contiuum robot:


## Env

### init
-No joint pos, vel, acc

### get_obs
-No joint pos, vel, acc

### reset
-random position of robot
    move motors and wait a bit for the physics to find a valid/stable start position


PPO TUNING (MUJOCO/ROBOTIK) - QUICK START
=========================================

1. VecNormalize (Wichtigster Schritt): done
   Umgebung unbedingt mit VecNormalize wrappen (norm_obs=True, norm_reward=True).

2. Netz-Architektur (policy_kwargs): done
   Von [64, 64] auf [256, 256] erhöhen. Mehr Kapazität für Physik-Modellierung.

3. Batch Size:
   Erhöhen auf 128 oder 256 für stabilere Gradienten.

4. Learning Rate:
   Start bei 3e-4, vorzugsweise mit linearem Decay (abfallend gegen 0).

5. Entropy Coefficient (ent_coef):
   Minimal erhöhen (z.B. 0.001), um Exploration zu erzwingen und Stillstand zu vermeiden.

6. n_steps:
   Zwischen 2048 und 4096 belassen für ausreichend Daten pro Update. -> wenn man n_envs erhöht  reduzeiren 





ZUM EVALUIEREN!!!! UND TRANIEREN 


- CALLBACK FUNKTION ODER ALLGMEIN CALLBACK BENUTZEN 

- Wichtige Metriken: 
Average Cumulative Reward (Return), Success Rate, Sample Efficiency
-> SB3 Monitor benutzen, teilweise müssen wir dazu noch was in die Env implementieren




TODO:

Seed rasunehmen!


check_env auf eine reduzieren

weitere rewards fct 

customcallback fct 

gute episoden speichern 

Episoden length nachschauen 

env bulletproof

hard terminated sachen 

rendern methode vibe code check

