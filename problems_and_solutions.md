-planerer roboter sehr schlecht beim trianing, da Ziele nicht erreichbar
    max range wird falsch berechnet, weil floor etc auch in die bodies zählen

- Render (planare roboter)
haben sehr gute modelle auf dem papier trainiert, die aber kompletter schrott waren später beim betrachten
    Die Rendermethode hat ein falschen Env gestartet ohne die Normalisierung von unseren VecEnvs

- im 3DOF arm hebt sich nicht wirklich in die Luft

- Roboterarm lernt sehr langsam, bzw nähert sich nicht wirklich gut an
    truncaten, wenn er sich nicht dem ziel nähert

- Roboter fährt in die Joint Limits
    punishment für nähe der joint limits über e-funktion

- Halten des Roboters in Position
    reward realtiv groß für steps im goal space

- kleine Goal spaces werden nicht gut angefahren
    distance reward scaliert mit dem -log, sehr hohe steigung bei kleinen distanzen
    distance wird als vektor auch in observation space übergeben
    goal distance kontinuirlich kleiner machen für besseres training?
    actions runterskalieren? 
        nicht mehr nur die aktion angucken, sondern das delta der actions betrachten
    torque doch mal vernüftig von den motoren einstellen?

- Singularität über dem roboter fürt zu problemen beim trainieren und beim rendern
    hohe geschwindigkeiten punishen?
    festlegen das 20 prozent der train runs eher über der base liegen
    vllt ist es doch keine singuöarität?
    winkel von ca 90 grad von joint 12 punishen oder gelenk nur von 0 bis 90 gehen lassen?
    geometrie des roboters hat ihm nicht erluabt diesen punkt zu erreichen

- roboter hat zu doll angst in den boden zu crashen
    soft punishment fpr nähe zum boden?

- roboter lässt sich auf joint limit ausruhen
    punishment hoch und joint lockerer machen, dass es zum crash kommen würde
    gelöst durch richtige rewards und punishment fpr joint limits

- wirkliches trackking und nachvollziehenen des verhaltens schwer#
    rewards plotten + video

- testset für roboter resulatate anders bei render mode
    mj_Data_reset resettet auch mujoco env

- reward hacking
    max steps kleiner machen, damit reward hacking sihc nicht lohnt
