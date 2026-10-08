import argparse
import json
import math
import os
from pathlib import Path
import time
import yaml


def apply_control_confirmed(client, vehicle, control):
    import carla
    responses = client.apply_batch_sync(
        [carla.command.ApplyVehicleControl(vehicle.id, control)], False)
    if len(responses) != 1 or responses[0].has_error():
        detail = responses[0].error if responses else "no response"
        raise RuntimeError("Control submission failed: {}".format(detail))


def read_config(path):
    with open(path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    for name in ("fixed_delta_seconds", "max_substep_delta_time", "tire_friction"):
        if not math.isfinite(config[name]) or config[name] <= 0:
            raise ValueError('Invalid positive parameter: {}'.format(name))
    if not isinstance(config["max_substeps"], int) or config["max_substeps"] < 1:
        raise ValueError('Positive max_substeps required')
    if config["fixed_delta_seconds"] > config["max_substep_delta_time"] * config["max_substeps"]:
        raise ValueError('Insufficient physics substep duration')
    if not isinstance(config["spawn_index"], int) or config["spawn_index"] < 0:
        raise ValueError('Nonnegative spawn_index required')
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weather", choices=("ClearNoon", "MidRainyNoon", "HardRainNoon"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).with_name("environment.yaml"))
    parser.add_argument("--steps", type=int, default=0,
                        help='Tick limit; 0=unlimited')
    parser.add_argument("--navigate", action="store_true", help="Enable BehaviorAgent navigation")
    parser.add_argument("--inspect-state", action="store_true", help="Print state and action measurements")
    parser.add_argument("--record", type=Path, metavar="DIRECTORY", help='Episode output directory')
    parser.add_argument("--state-every", type=int, default=20, help='State interval; default 20')
    parser.add_argument("--carla-root", type=Path,
                        default=Path(os.environ.get("CARLA_ROOT", "../CARLA_0.9.15")))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--destinations", type=int, default=1,
                        help='Destination count; 0=unlimited')
    args = parser.parse_args()
    if args.steps < 0:
        parser.error("--steps must be nonnegative")
    if args.destinations < 0:
        parser.error("--destinations must be nonnegative")
    if args.state_every < 1:
        parser.error("--state-every must be positive")
    config = read_config(args.config)
    if args.weather: config["weather"] = args.weather
    if args.navigate:
        from navigation import Navigation, load_behavior_agent
        behavior_agent = load_behavior_agent(args.carla_root)
    import carla

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    print("Client/server: {}/{}".format(client.get_client_version(),
                                       client.get_server_version()), flush=True)
    world = client.get_world()
    if world.get_settings().synchronous_mode:
        raise RuntimeError('Server is already synchronous')
    if world.get_actors().filter("vehicle.*") or world.get_actors().filter("walker.pedestrian.*"):
        raise RuntimeError('Server contains vehicles/pedestrians')
    if world.get_map().name.rsplit("/", 1)[-1] != config["map"]:
        print('Loading {}'.format(config["map"]), flush=True)
        world = client.load_world(config["map"])
    if world.get_actors().filter("static.trigger.friction"):
        raise RuntimeError('Existing friction triggers detected')

    original_settings = world.get_settings()
    original_weather = world.get_weather()
    spectator = world.get_spectator()
    original_view = spectator.get_transform()
    vehicle = None
    navigation = None
    monitor = None
    recorder = None
    end_reason = "error"
    error = None
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = config["fixed_delta_seconds"]
        settings.substepping = True
        settings.max_substep_delta_time = config["max_substep_delta_time"]
        settings.max_substeps = config["max_substeps"]
        settings.no_rendering_mode = False
        world.apply_settings(settings)
        weather = getattr(carla.WeatherParameters, config["weather"])
        world.set_weather(weather)

        spawns = world.get_map().get_spawn_points()
        if config["spawn_index"] >= len(spawns):
            raise ValueError('Spawn index exceeds {} points'.format(len(spawns)))
        blueprint = world.get_blueprint_library().find(config["vehicle"])
        blueprint.set_attribute("role_name", "low_friction_ego")
        if blueprint.has_attribute("color"):
            blueprint.set_attribute("color", "255,255,255")
        if blueprint.has_attribute("terramechanics"):
            blueprint.set_attribute("terramechanics", "false")
        vehicle = world.try_spawn_actor(blueprint, spawns[config["spawn_index"]])
        if vehicle is None:
            raise RuntimeError('Fixed spawn point is occupied')
        physics = vehicle.get_physics_control()
        before = [wheel.tire_friction for wheel in physics.wheels]
        if len(before) != 4 or any(config["tire_friction"] >= value for value in before):
            raise RuntimeError('Invalid wheel friction: {}'.format(before))
        wheels = physics.wheels
        for wheel in wheels:
            wheel.tire_friction = config["tire_friction"]
        physics.wheels = wheels
        vehicle.apply_physics_control(physics)
        apply_control_confirmed(client, vehicle, carla.VehicleControl(brake=1.0))

        for _ in range(20):
            world.tick()
        location = vehicle.get_location()
        spectator.set_transform(carla.Transform(
            carla.Location(x=location.x, y=location.y, z=location.z + 12),
            carla.Rotation(pitch=-90, yaw=vehicle.get_transform().rotation.yaw)))

        actual = world.get_settings()
        friction = [wheel.tire_friction for wheel in vehicle.get_physics_control().wheels]
        checks = {
            "map": world.get_map().name.rsplit("/", 1)[-1] == config["map"],
            "vehicle": vehicle.type_id == config["vehicle"],
            "synchronous_mode": actual.synchronous_mode,
            "fixed_delta_seconds": math.isclose(actual.fixed_delta_seconds, config["fixed_delta_seconds"], abs_tol=1e-6),
            "substepping": actual.substepping,
            "max_substeps": actual.max_substeps == config["max_substeps"],
            "max_substep_delta_time": math.isclose(actual.max_substep_delta_time, config["max_substep_delta_time"], abs_tol=1e-6),
            "wheel_friction": len(friction) == 4 and all(math.isclose(v, config["tire_friction"], abs_tol=1e-6) for v in friction),
        }
        weather_fields = ("cloudiness", "precipitation", "precipitation_deposits",
                          "wind_intensity", "sun_azimuth_angle", "sun_altitude_angle",
                          "fog_density", "fog_distance", "wetness", "fog_falloff",
                          "scattering_intensity", "mie_scattering_scale", "rayleigh_scattering_scale")
        actual_weather = world.get_weather()
        checks["weather"] = all(math.isclose(getattr(actual_weather, key), getattr(weather, key), abs_tol=1e-4)
                                for key in weather_fields)
        print(json.dumps({"config": config, "original_wheel_friction": before,
                          "actual_wheel_friction": friction, "checks": checks}, indent=2), flush=True)
        if not all(checks.values()):
            raise RuntimeError("Environment readback verification failed")
        print('PASS: environment configured', flush=True)
        if args.navigate:
            navigation = Navigation(world, vehicle, behavior_agent, config["fixed_delta_seconds"],
                                    args.seed, args.destinations)
        else:
            print("Stationary check: vehicle stays braked.", flush=True)
        count = 0
        previous = world.get_snapshot()
        if args.record:
            from recording import EpisodeRecorder
            recorder = EpisodeRecorder(args.record, dict(
                config=config, client_version=client.get_client_version(), server_version=client.get_server_version(),
                actual_wheel_friction=friction, environment_checks=checks, seed=args.seed,
                controller="BehaviorAgent:cautious" if args.navigate else "stationary_brake",
                initial_route=navigation.route_info if navigation else None,
                requested_destinations=args.destinations, step_limit=args.steps), vehicle, previous)
        if args.inspect_state:
            from state_monitor import StateMonitor
            monitor = StateMonitor(vehicle, previous, args.state_every)
        while args.steps == 0 or count < args.steps:
            started = time.monotonic()
            if navigation is not None:
                if navigation.finished:
                    break
                requested_control = navigation.before_tick()
            else:
                requested_control = carla.VehicleControl(brake=1.0)
            apply_control_confirmed(client, vehicle, requested_control)
            frame = world.tick()
            snapshot = world.get_snapshot()
            if frame != previous.frame + 1 or snapshot.frame != frame:
                raise RuntimeError('Unexpected frame advance')
            if not math.isclose(snapshot.timestamp.delta_seconds, config["fixed_delta_seconds"], abs_tol=1e-6):
                raise RuntimeError("Simulation timestep changed")
            previous = snapshot
            count += 1
            if monitor is not None:
                monitor.update(snapshot, requested_control)
            if recorder is not None:
                recorder.append(snapshot, requested_control, navigation.completed + 1 if navigation else 0)
            if navigation is not None:
                navigation.after_tick(count)
            if count % 100 == 0:
                print('PASS: ticks={} dt={:.3f}s'.format(count, snapshot.timestamp.delta_seconds), flush=True)
            time.sleep(max(0, config["fixed_delta_seconds"] - (time.monotonic() - started)))
        if navigation is not None and not navigation.finished:
            print('Navigation reached step limit', flush=True)
        end_reason = "destination_complete" if navigation is not None and navigation.finished else "step_limit"
    except KeyboardInterrupt:
        end_reason = "user_interrupt"
        print("Stopping.", flush=True)
    except Exception as exc:
        error = "{}: {}".format(type(exc).__name__, exc)
        raise
    finally:
        if monitor is not None:
            monitor.summary()
        cleanup = []
        if recorder is not None:
            cleanup.append(("recording", lambda: recorder.close(end_reason, error)))
        if navigation is not None:
            cleanup.append(("navigation sensors", navigation.close))
        if vehicle is not None:
            cleanup.append(("vehicle", vehicle.destroy))
        cleanup.extend([("weather", lambda: world.set_weather(original_weather)),
                        ("spectator", lambda: spectator.set_transform(original_view)),
                        ("world settings", lambda: world.apply_settings(original_settings))])
        for label, action in cleanup:
            try:
                action()
            except Exception as exc:
                print("Cleanup failed ({}): {}".format(label, exc), flush=True)
        print('Cleanup finished', flush=True)


if __name__ == "__main__":
    main()
