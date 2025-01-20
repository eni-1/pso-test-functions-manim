from manim import *
import numpy as np
import random
from dataclasses import dataclass
from typing import List, Tuple, Callable
from scipy.interpolate import RegularGridInterpolator


@dataclass
class ParticleState:
    position: np.ndarray
    score: float
    is_best: bool


@dataclass
class IterationState:
    particles: List[ParticleState]
    global_best_position: np.ndarray
    global_best_score: float


class Particle:
    def __init__(self, dimensions: int, bounds: Tuple[float, float], velocity_bounds: Tuple[float, float] = None):
        self.position = np.array([random.uniform(bounds[0], bounds[1]) for _ in range(dimensions)])

        if velocity_bounds is None:
            velocity_bounds = (bounds[0] * 0.1, bounds[1] * 0.1)
        self.velocity = np.array([random.uniform(velocity_bounds[0], velocity_bounds[1]) for _ in range(dimensions)])
        self.best_position = self.position.copy()
        self.best_score = float('inf')

    def update(self, objective_fn: Callable, global_best_position: np.ndarray, params: dict) -> float:
        w = params.get('w', 0.729)
        c1 = params.get('c1', 1.49445)
        c2 = params.get('c2', 1.49445)
        max_velocity = params.get('max_velocity', 30.0)
        bounds = params.get('bounds', (-500, 500))
        r1, r2 = np.random.random(2)
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best_position - self.position)

        self.velocity = w * self.velocity + cognitive + social
        self.velocity = np.clip(self.velocity, -max_velocity, max_velocity)
        self.position = np.clip(self.position + self.velocity, bounds[0], bounds[1])

        score = objective_fn(self.position)
        if score < self.best_score:
            self.best_score = score
            self.best_position = self.position.copy()

        return score


def optimize(objective_fn: Callable,
             dimensions: int,
             bounds: Tuple[float, float],
             n_trials: int = 100,
             n_particles: int = 40,
             max_iterations: int = 100,
             convergence_threshold: float = 1e-8,
             patience: int = 15,
             min_iterations: int = 20,
             pso_params: dict = None) -> List[IterationState]:
    if pso_params is None:
        pso_params = {
            'w': 0.729,
            'c1': 1.49445,
            'c2': 1.49445,
            'max_velocity': 30.0,
            'bounds': bounds
        }

    best_run_states = None
    best_final_score = float('inf')

    for trial in range(n_trials):
        iteration_states = []
        particles = [Particle(dimensions, bounds) for _ in range(n_particles)]
        global_best_score = float('inf')
        global_best_position = None
        iterations_without_improvement = 0
        previous_best_score = float('inf')

        for iteration in range(max_iterations):
            current_particle_states = []

            for particle in particles:
                current_score = particle.update(
                    objective_fn,
                    global_best_position if global_best_position is not None else particle.position,
                    pso_params
                )

                if current_score < global_best_score:
                    global_best_score = current_score
                    global_best_position = particle.position.copy()

                current_particle_states.append(ParticleState(
                    position=particle.position.copy(),
                    score=current_score,
                    is_best=(current_score == global_best_score)
                ))

            iteration_states.append(IterationState(
                particles=current_particle_states,
                global_best_position=global_best_position.copy(),
                global_best_score=global_best_score
            ))

            if iteration >= min_iterations:
                improvement = previous_best_score - global_best_score

                if improvement < convergence_threshold:
                    iterations_without_improvement += 1
                else:
                    iterations_without_improvement = 0

                if iterations_without_improvement >= patience:
                    print(
                        f"Trial {trial + 1}: Converged after {iteration + 1} iterations with score {global_best_score:.10f}")
                    break

                previous_best_score = global_best_score

        if global_best_score < best_final_score:
            best_final_score = global_best_score
            best_run_states = iteration_states
            print(f"Trial {trial + 1}: Found better solution with score {best_final_score:.10f}")

    print(f"\nBest solution found: Score = {best_final_score:.10f}")
    return best_run_states


def schwefel(x: np.ndarray) -> float:
    x_val, y_val = x[0], x[1]
    return 418.9829 * 2 - (x_val * np.sin(np.sqrt(abs(x_val))) +
                           y_val * np.sin(np.sqrt(abs(y_val))))


def rosenbrock(x: np.ndarray) -> float:
    return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)


class Visualization(ThreeDScene):
    def __init__(self,
                 objective_fn: Callable = None,
                 bounds: Tuple[float, float] = None,
                 pso_params: dict = None,
                 surface_resolution: int = 100,
                 z_range: List[float] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.objective_fn = objective_fn
        self.bounds = bounds
        self.pso_params = pso_params if pso_params is not None else {}
        self.surface_resolution = surface_resolution
        self.z_range = z_range if z_range is not None else [0, 2000, 500]
        self.pregen_surface_mesh()

    def pregen_surface_mesh(self):
        self.x_vals = np.linspace(self.bounds[0], self.bounds[1], self.surface_resolution)
        self.y_vals = np.linspace(self.bounds[0], self.bounds[1], self.surface_resolution)
        x_mesh, y_mesh = np.meshgrid(self.x_vals, self.y_vals)
        positions = np.stack([x_mesh.flatten(), y_mesh.flatten()], axis=1)
        z_values = np.array([float(self.objective_fn(pos)) for pos in positions])
        self.z_mesh = z_values.reshape(x_mesh.shape)
        self.interpolator = RegularGridInterpolator(
            (self.x_vals, self.y_vals),
            self.z_mesh.T,
            method='linear',
            bounds_error=False,
            fill_value=None
        )

    def get_surface_position(self, x: float, y: float) -> np.ndarray:
        x = np.clip(x, self.bounds[0], self.bounds[1])
        y = np.clip(y, self.bounds[0], self.bounds[1])

        x_idx = np.abs(self.x_vals - x).argmin()
        y_idx = np.abs(self.y_vals - y).argmin()

        z = float(self.interpolator([x, y]))
        return np.array([x, y, z])

    def make_surface(self, axes):
        surface = Surface(
            lambda u, v: axes.c2p(
                self.x_vals[int(u * (self.surface_resolution - 1))],
                self.y_vals[int(v * (self.surface_resolution - 1))],
                self.z_mesh[int(v * (self.surface_resolution - 1)),
                int(u * (self.surface_resolution - 1))]
            ),
            u_range=[0, 1],
            v_range=[0, 1],
            resolution=(self.surface_resolution, self.surface_resolution),
            should_make_jagged=True,
            fill_opacity=0.3,
            stroke_opacity=0.2,
            stroke_width=0.3,
            stroke_color=WHITE,
            shade_in_3d=True
        )

        surface.set_fill_by_value(
            axes=axes,
            colors=[
                (PURPLE_E, self.z_mesh.min()),
                (BLUE_D, self.z_mesh.min() * 0.7),
                (BLUE_C, self.z_mesh.min() * 0.2),
                (TEAL, 0),
                (YELLOW_D, self.z_mesh.max() * 0.2),
                (ORANGE, self.z_mesh.max() * 0.7),
                (RED, self.z_mesh.max())
            ],
            axis=2
        )
        return surface

    def particle_visuals(self, particle_pos, axes):
        surface_pos = self.get_surface_position(*particle_pos)
        manim_pos = axes.c2p(*surface_pos)

        dot = Sphere(radius=0.04).move_to(manim_pos)
        dot.set_color(BLUE_A)
        glow = Sphere(radius=0.06).move_to(manim_pos)
        glow.set_color(BLUE_A)
        glow.set_opacity(0.3)
        particle = VGroup(dot, glow)
        shadow_pos = [surface_pos[0], surface_pos[1], self.z_range[0]]
        shadow = Dot(
            point=axes.c2p(*shadow_pos),
            color=BLUE_A,
            fill_opacity=0.3
        )

        return particle, shadow

    def construct(self):
        print("Starting visualization...")
        iteration_states = optimize(
            objective_fn=self.objective_fn,
            dimensions=2,
            bounds=self.bounds,
            pso_params=self.pso_params
        )

        self.camera.background_color = BLACK

        axes = ThreeDAxes(
            x_range=[self.bounds[0], self.bounds[1], (self.bounds[1] - self.bounds[0]) / 5],
            y_range=[self.bounds[0], self.bounds[1], (self.bounds[1] - self.bounds[0]) / 5],
            z_range=self.z_range,
            x_length=8,
            y_length=8,
            z_length=4,
            axis_config={"include_tip": False, "include_numbers": False}
        )

        surface = self.make_surface(axes)

        particle_dots = VGroup()
        particle_shadows = VGroup()

        for particle_state in iteration_states[0].particles:
            particle, shadow = self.particle_visuals(
                particle_state.position,
                axes
            )
            particle_dots.add(particle)
            particle_shadows.add(shadow)

        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES, zoom=0.8, frame_center=[0, 0, 2])
        self.begin_ambient_camera_rotation(rate=0.1)
        self.play(Create(axes), run_time=1)
        self.play(Create(surface), run_time=1.5)
        self.play(
            Create(particle_dots),
            Create(particle_shadows),
            run_time=1
        )

        for iteration_state in iteration_states:
            animations = []

            for i, (particle_state, dot_group, shadow) in enumerate(
                    zip(iteration_state.particles, particle_dots, particle_shadows)
            ):
                surface_pos = self.get_surface_position(*particle_state.position)
                manim_pos = axes.c2p(*surface_pos)
                animations.append(dot_group.animate.move_to(manim_pos))
                shadow_pos = [surface_pos[0], surface_pos[1], self.z_range[0]]
                animations.append(
                    shadow.animate.move_to(axes.c2p(*shadow_pos))
                )

            self.play(
                *animations,
                run_time=0.5,
                rate_func=smooth
            )

        final_state = iteration_states[-1]
        result_text = Text(
            f"({final_state.global_best_position[0]:.4f}, {final_state.global_best_position[1]:.4f})",
            font_size=24
        ).to_edge(DOWN)

        self.add_fixed_in_frame_mobjects(result_text)
        self.play(FadeIn(result_text))
        self.wait(2)


class SchwefelPSO(Visualization):
    def __init__(self, **kwargs):
        super().__init__(
            objective_fn=schwefel,
            bounds=(-500, 500),
            pso_params={
                'w': 0.729,
                'c1': 1.49445,
                'c2': 1.49445,
                'max_velocity': 30.0,
                'bounds': (-500, 500)
            },
            **kwargs
        )


class RosenbrockPSO(Visualization):
    def __init__(self, **kwargs):
        super().__init__(
            objective_fn=rosenbrock,
            bounds=(-2, 2),
            pso_params={
                'w': 0.729,
                'c1': 1.49445,
                'c2': 1.49445,
                'max_velocity': 0.5,
                'bounds': (-2, 2)
            },
            **kwargs
        )

# to use run 'manim manimpso.py RosenbrockPSO'
