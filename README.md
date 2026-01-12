## Question 1 — First Contact with Snellius (Cluster Access)

### (a) SSH-verbinding en login node

Ik heb verbinding gemaakt met Snellius met het volgende commando:

```bash
ssh scur2292@snellius.surf.nl
```
Na het inloggen kwam ik terecht op login node int5, wat zichtbaar was in de shell prompt.
\![SSH login op int5](![SSH login op int5](MLOps_2026/assets/image.png)
)

###  (b) Problemen en onduidelijkheden tijdens het verbinden

Toen ik voor het eerst probeerde in te loggen, lukte de SSH-handshake wel, maar werd de verbinding direct daarna weer gesloten. Dit gebeurde vlak nadat ik mijn SURF-wachtwoord had ingesteld.

Het gedrag dat ik zag was dat de server bereikbaar was, maar dat de loginsessie meteen werd beëindigd zonder een duidelijke foutmelding zoals "wrong password". Dit was verwarrend, omdat het leek alsof mijn SSH-configuratie fout was.

Uiteindelijk bleek dat mijn SURF-account nog niet volledig geactiveerd was in alle systemen. Nadat ik een tijdje had gewacht en het de volgende dag opnieuw probeerde, werkte de SSH-verbinding wel direct. Daarmee was duidelijk dat het probleem niet bij mijn instellingen lag, maar bij de account-propagatie aan de kant van SURF.

### (c) SSH-client, ervaring en voorzorgsmaatregelen

Ik heb de standaard OpenSSH-client gebruikt om verbinding te maken met Snellius.

Ik had vóór deze opdracht slechts beperkte ervaring met SSH (voornamelijk simpele ssh user@host commando’s). Daarom heb ik bewust eerst geprobeerd om met alleen een wachtwoord in te loggen, voordat ik met SSH-sleutels begon.

Om veelgemaakte fouten te vermijden heb ik:

De officiële hostnaam snellius.surf.nl gebruikt

Gecontroleerd dat ik mijn juiste SURF-gebruikersnaam scur2292 gebruikte

Eerst getest of inloggen met wachtwoord werkte voordat ik verder ging met key-based login

Hierdoor kon ik goed onderscheiden of problemen werden veroorzaakt door accountactivatie of door een fout in mijn SSH-configuratie.