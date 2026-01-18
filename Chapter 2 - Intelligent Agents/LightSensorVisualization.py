import pygame
import sys
from SimpleReflexAgent import SmartLightTimedReflexAgent

# Simple visualization of the Smart Light Timed Reflex Agent using Pygame

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 600

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)
DARK_GRAY = (40, 40, 40)
YELLOW = (255, 255, 0)
BRIGHT_YELLOW = (255, 255, 0)
BLUE = (100, 150, 255)
RED = (255, 100, 100)

# Create the screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Smart Light Timed Reflex Agent Visualization")

# Font for text
font_large = pygame.font.Font(None, 54)
font_medium = pygame.font.Font(None, 46)
font_small = pygame.font.Font(None, 38)

def draw_scene(screen, second, percept, agent, action):
    """Draw the visualization scene"""
    screen.fill(WHITE)
    
    # Draw floor/ground
    pygame.draw.line(screen, BLACK, (50, 500), (950, 500), 3)
    
    # Draw room background
    pygame.draw.rect(screen, (120, 120, 120), (50, 100, 900, 400), 0)
    
    # Draw light fixture (ceiling)
    light_x = 850
    light_y = 150
    
    # Light bulb - use agent's light state
    light_color = BRIGHT_YELLOW if agent.light_is_on else DARK_GRAY
    pygame.draw.circle(screen, light_color, (light_x, light_y), 30)
    pygame.draw.circle(screen, BLACK, (light_x, light_y), 30, 2)
    
    # Light fixture base
    pygame.draw.rect(screen, GRAY, (light_x - 15, light_y + 25, 30, 15))
    
    # Draw light rays if light is on
    if agent.light_is_on:
        ray_color = (255, 255, 150, 50)
        pygame.draw.polygon(screen, (255, 255, 150), [
            (light_x, light_y + 30),
            (light_x - 150, 500),
            (light_x + 150, 500)
        ])
    
    # Draw motion sensor (near the light)
    sensor_x = light_x - 80
    sensor_y = light_y + 100
    sensor_color = BLUE if percept[0] else GRAY
    pygame.draw.rect(screen, sensor_color, (sensor_x - 20, sensor_y - 20, 40, 40))
    pygame.draw.rect(screen, BLACK, (sensor_x - 20, sensor_y - 20, 40, 40), 2)
    pygame.draw.circle(screen, BLACK, (sensor_x, sensor_y), 5)
    
    # Get motion detected status
    motion_detected = percept[0]
    
    # Draw stick man (motion source) - only when motion detected
    if motion_detected:
        stickman_x = 300
        stickman_y = 350
        
        # Head (circle)
        pygame.draw.circle(screen, RED, (stickman_x, stickman_y - 40), 20)
        # Body (vertical line)
        pygame.draw.line(screen, RED, (stickman_x, stickman_y - 20), (stickman_x, stickman_y + 20), 3)
        # Left arm
        pygame.draw.line(screen, RED, (stickman_x, stickman_y - 10), (stickman_x - 25, stickman_y - 5), 3)
        # Right arm
        pygame.draw.line(screen, RED, (stickman_x, stickman_y - 10), (stickman_x + 25, stickman_y - 5), 3)
        # Left leg
        pygame.draw.line(screen, RED, (stickman_x, stickman_y + 20), (stickman_x - 15, stickman_y + 50), 3)
        # Right leg
        pygame.draw.line(screen, RED, (stickman_x, stickman_y + 20), (stickman_x + 15, stickman_y + 50), 3)
        
        # Draw motion sensing line from sensor to stick man
        pygame.draw.line(screen, RED, (sensor_x, sensor_y), (stickman_x, stickman_y - 40), 3)
    
    # Draw information panel
    info_y = 50
    
    # Title
    title_text = font_large.render("Smart Light Agent Simulation", True, BLACK)
    screen.blit(title_text, (50, info_y))
    
    # Time display
    time_text = font_medium.render(f"Time: Second {second:02d}", True, BLACK)
    screen.blit(time_text, (50, info_y + 60))
    
    # Percept display
    motion_str = "MOTION DETECTED" if percept[0] else "No Motion"
    light_str = percept[1]
    percept_text = font_small.render(f"Percept: {motion_str}, {light_str}", True, BLACK)
    screen.blit(percept_text, (50, info_y + 100))
    
    # Timer display
    timer_text = font_small.render(f"No-Motion Timer: {agent.no_motion_timer:02d}s", True, BLACK)
    screen.blit(timer_text, (50, info_y + 130))
    
    # Action display
    display_action = action
    if not agent.light_is_on and not percept[0] and "Turn Light Off" not in action:
        display_action = "Do Nothing"
    
    action_color = BRIGHT_YELLOW if "Turn Light On" in display_action else (RED if "Turn Light Off" in display_action else BLUE)
    action_text = font_medium.render(f"Action: {display_action}", True, action_color)
    screen.blit(action_text, (50, info_y + 170))
    
    # Status display
    light_status = "ON" if agent.light_is_on else "OFF"
    status_color = BRIGHT_YELLOW if agent.light_is_on else DARK_GRAY
    status_text = font_medium.render(f"Light Status: {light_status}", True, status_color)
    screen.blit(status_text, (550, info_y + 100))

def main():
    clock = pygame.time.Clock()
    agent = SmartLightTimedReflexAgent()
    
    running = True
    current_second = 0
    display_fps = 5  # Display 5 frames per second
    
    motion_active = False
    
    percept = (False, "Dark") # Initial percept of no motion, no light
    
    # Title
    title_text = "Smart Light Agent Simulation - Press SPACE to toggle motion"
    pygame.display.set_caption(title_text)
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # Toggle motion when space is pressed
                    motion_active = not motion_active
        
        # Update agent state once per frame (which is 1 simulation second at 5 FPS)
        if motion_active:
            percept = (True, "Dark")
        else:
            percept = (False, "Light")
        
        action = agent.act(percept)
        current_second += 1
        
        draw_scene(screen, current_second, percept, agent, action)
        
        pygame.display.flip()
        clock.tick(display_fps)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
