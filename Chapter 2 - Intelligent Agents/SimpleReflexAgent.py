class SmartLightTimedReflexAgent:
    def __init__(self): #self is the instance of the class (like this in Java/C#)
        self.no_motion_timer = 0  # seconds since last motion
        self.light_duration = 20  # seconds to keep light on after no motion
        self.light_is_on = False  # Track light state separately
        
    def act(self, percept):
        motion_detected, ambient_light = percept # separates the percept tuple into two variables

        if motion_detected:
            self.no_motion_timer = 0
            if ambient_light == "Dark":
                self.light_is_on = True
                return "Turn Light On"
            else:
                return "Do Nothing"

        # No motion detected
        self.no_motion_timer = min(self.no_motion_timer + 1, self.light_duration + 1)

        if self.no_motion_timer >= self.light_duration and self.light_is_on:
            self.light_is_on = False
            return "Turn Light Off"
        else:
            return "Do Nothing"
        
    # Runs a simulation of the agent's behavior over the number of percepts provided
    def run_simulation(agent, percepts):
        for second, percept in enumerate(percepts):
            action = agent.act(percept)
            print(f"Second {second:02d} | Percept: {percept} | Timer: {agent.no_motion_timer:02d} | Action: {action}")

# main function
def main():
    agent = SmartLightTimedReflexAgent()
    percepts = [
        (True, "Dark"),   # Motion detected, dark
        (True, "Light"),  # Motion detected, light
        (True, "Light"),  # Motion detected, already lit
    ] + [(False, "Light")] * 22  # No motion for 62 seconds

    SmartLightTimedReflexAgent.run_simulation(agent, percepts)

if __name__ == "__main__":
    main()
    