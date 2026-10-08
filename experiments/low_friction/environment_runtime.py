def apply_control_confirmed(client, vehicle, control):
    import carla
    responses = client.apply_batch_sync(
        [carla.command.ApplyVehicleControl(vehicle.id, control)], False)
    if len(responses) != 1 or responses[0].has_error():
        detail = responses[0].error if responses else "no response"
        raise RuntimeError("Control submission failed: {}".format(detail))
