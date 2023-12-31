import os
import pygame
import math
from dataclasses import dataclass
from typing import List
import torch

# Define the screen dimensions
SCREEN_SIZE = 800
N_FRAMES = 1000
SAVE_DIR = "outputs/"
# Define colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)

@dataclass
class Birdoid:
    x: float
    y: float
    angle: float  # Angle in radians

class Boids:
    def __init__(
        self,
        n_agents: int,
        screen_size: float,
        screen_margin: int,
        protected_range: int,
        visual_range: int,
        avoidfactor: float,
        matchingfactor: float,
        centeringfactor: float,
        turnfactor: float,
        minspeed: float,
        maxspeed: float,
    ):
        self.n_agents = n_agents
        self.screen_size = screen_size
        self.screen_margin = screen_margin
        self.protected_range = protected_range
        self.visual_range = visual_range
        self.avoidfactor = avoidfactor
        self.matchingfactor = matchingfactor
        self.centeringfactor = centeringfactor
        self.turnfactor = turnfactor
        self.minspeed = minspeed
        self.maxspeed = maxspeed
        self.position = screen_size * torch.rand(n_agents, 2)  # x, y
        self.velocity = torch.randn(n_agents, 2)  # vx, vy
        # atan(vy / vx) is the angle
        pass

    def step(self, dt: float):
        # compute distances
        diff = self.position[:, None, ...] - self.position[None, ...]  # 1, N, 2 - N, 1, 2 = N, N, 2
        distances = torch.sqrt(diff.pow(2).sum(dim=2))

        # separation
        in_protected_range = distances < self.protected_range  # N, N
        new_velocity = self.velocity + self.avoidfactor * (diff * in_protected_range[..., None].float()).sum(1)

        in_visual_range = (~in_protected_range) & (distances < self.visual_range)  # N, N
        in_visual_range_fp32 = in_visual_range.float()
        count_in_visual_range = in_visual_range_fp32.sum(1, keepdim=True)
        # alignment
        new_velocity += torch.nan_to_num(self.matchingfactor * (in_visual_range_fp32 @ self.velocity) / count_in_visual_range)

        # cohesion
        visible_com = torch.nan_to_num(torch.mm(in_visual_range_fp32, self.position) / count_in_visual_range)
        new_velocity += self.centeringfactor * (visible_com - self.position)

        # boundaries
        left_margin = self.position[..., 0] < self.screen_margin
        right_margin = self.position[..., 0] > (self.screen_size - self.screen_margin)
        top_margin = self.position[..., 1] < self.screen_margin
        bottom_margin = self.position[..., 1] > (self.screen_size - self.screen_margin)
        new_velocity[left_margin, 0] += self.turnfactor
        new_velocity[right_margin, 0] -= self.turnfactor
        new_velocity[top_margin, 1] += self.turnfactor
        new_velocity[bottom_margin, 1] -= self.turnfactor
        # speed limit
        speed = new_velocity.norm(dim=1, keepdim=True)
        new_velocity = (new_velocity / speed) * torch.clamp(speed, min=self.minspeed, max=self.maxspeed)
        self.velocity = new_velocity
        self.position += self.velocity * dt

    def render(self, screen) -> None:
        # angle1 = torch.atan(self.velocity[..., 1] / self.velocity[..., 0])
        # angle1_x, angle1_y = torch.cos(angle1), torch.sin(angle1)
        # angle2 = angle1 + (2 * math.pi / 3)
        # angle3 = angle1 - (2 * math.pi / 3)
        # # Calculate triangle vertices
        # x1, y1 = self.position[..., 0] + 15 * angle1_x, self.position[..., 1] + 15 * angle1_y
        # x2, y2 = self.position[..., 0] + 15 * torch.cos(angle2), self.position[..., 1] + 15 * torch.sin(angle2)
        # x3, y3 = self.position[..., 0] + 15 * torch.cos(angle3), self.position[..., 1] + 15 * torch.sin(angle3)
        # x4, y4 = self.position[..., 0] + 10 * angle1_x, self.position[..., 1] + 10 * angle1_y

        # Draw polygons and lines
        normalized_velocity = self.velocity / self.velocity.norm(dim=1, keepdim=True)
        for agent_idx in range(self.n_agents):
            x = self.position[agent_idx, 0].item()
            y = self.position[agent_idx, 1].item()
            x_other = self.position[agent_idx, 0].item() + 10 * normalized_velocity[agent_idx, 0].item()
            y_other = self.position[agent_idx, 1].item() + 10 * normalized_velocity[agent_idx, 1].item()
            pygame.draw.circle(screen, WHITE, (x, y), 4, 2)
            pygame.draw.line(screen, RED, (x, y), (x_other, y_other), 4)

# Main function
def main():
    os.makedirs(os.path.join(SAVE_DIR, "frames/"), exist_ok=True)
    pygame.init()

    # Create the screen
    screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
    pygame.display.set_caption("Boids Algorithm")

    # Create some sample birdoids
    boids = Boids(
        screen_size=SCREEN_SIZE,
        screen_margin=SCREEN_SIZE // 10,
        n_agents=1000,
        protected_range=5,
        visual_range=50,
        avoidfactor=0.05,
        matchingfactor=0.05,
        centeringfactor=0.000005,
        turnfactor=0.2,
        minspeed=2,
        maxspeed=3,
    )

    clock = pygame.time.Clock()

    running = True
    frame_idx = 0
    while running and (frame_idx < N_FRAMES):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Clear the screen
        screen.fill((0, 0, 0))

        # Render birdoids
        boids.render(screen)
        boids.step(dt=1)
        # Update the display
        pygame.display.flip()
        pygame.image.save(screen, os.path.join(SAVE_DIR, "frames", f"frame_{frame_idx:04d}.png"))

        # Cap the frame rate
        clock.tick(60)
        frame_idx += 1

    pygame.quit()

if __name__ == "__main__":
    main()
