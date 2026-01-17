# Improvements?

Steps passed/max steps als input ins netz?
we could track the real energy by torque* w(winkelgeschw.)
So layer stacken für time awarnes, als vergangene werte werden mitgetracked




# Changes we got to make later on for the contiuum robot:


## Env

### init
-No joint pos, vel, acc

### get_obs
-No joint pos, vel, acc

### reset
-random position of robot
    move motors and wait a bit for the physics to find a valid/stable start position
