# Method details

## Direct objective

The fixed TCP proposal is held over H=10 predicted steps, dt=0.05 s; the residual policy is evaluated at each predicted state. Only policy parameters receive gradients.

`L = L_position + L_beta + L_road + L_low_speed + 20 L_lane`.

Position is the squared 0.5-second GNSS-derived endpoint residual to TCP's first waypoint, scaled by 1 m. Road uses squared actor-point boundary violations in a same-direction Driving envelope. Lane uses the worst of four vehicle footprint corners, a 0.1 m interior margin, 1 m distance scale and mean over the horizon.

Sideslip is `atan2(vy, max(vx,0.5))` in regularized training mode, divided by 5 degrees and squared. Eligibility is fixed from the initial observed `vx >= 0.5`; predicted slowing does not remove a frame from the loss. The low-speed guard is the mean squared `relu(0.5-vx)/0.5` over eligible windows.

The geometry implementation assumes CARLA actor x-forward/y-right and a TCP GNSS mounting offset (-1.4 m, 0). Simulator-frame and coordinate tests support the implementation.

## SAC reward

Per real transition (dt=0.05 s), the implementation adds:

| Term | Contribution |
|---|---|
| New progress | Newly reached route distance in meters; high-water mark prevents back-and-forth reward farming |
| Time | `-0.02 * dt` |
| Local TCP lateral error | `-0.5 * lateral_error² * dt` |
| Road departure | `-4 * footprint_outside_depth² * dt`, only where geometry is covered |
| Sideslip | `-0.05 * min((beta / 5deg)², 36) * dt`, with observed vx eligibility |
| Residual magnitude | `-0.01 * mean(normalized_residual²) * dt` |
| Residual change | `-0.02 * mean((normalized_residual - previous_residual)²)` |
| Collision | `-300` in the final training configuration |
| Goal | `+10` on real route completion |

These rewards are not the same as the direct-policy loss. In particular, the SAC local lateral term is not a full 0.5-second endpoint objective. Collision weight began at -30 historically and was raised to -300.

SAC uses gamma=0.99, tau=0.005, batch size 64, learning rate 3e-4, entropy temperature initialized to 0.02 and automatic entropy tuning. Real replay capacity is 100,000; model replay capacity is 4,096; normal mixed batches contain 48 real plus 16 synthetic transitions. Updates are scheduled every four newly stored real transitions after warmup, and only gate-valid states contribute actor updates.

## Synthetic branches

One deterministic Transformer predicts one next state. The current TCP base command and world-space reference/lane geometry are held for the branch. Predicted next observations update physical history and transform the fixed geometry into the predicted actor frame. No differentiation through dynamics is used by SAC.

Accepted prediction ranges are `0.5 <= vx <= 12 m/s`, `|vy| <= 6 m/s`, `|r| <= 2 rad/s`, planar displacement <=1 m and `|Δyaw| <=0.15 rad`. Branches also require finite outputs, covered predicted footprint and remaining route distance >=10 m. These broad guards are not probabilistic uncertainty estimates.

Synthetic transitions are nonterminal, model-horizon truncated and bootstrap their target values. Collision/goal terms are zero because the model does not predict those events; this is not evidence that a synthetic action is collision-free. Real and model pools are not silently merged.

## Deployment and original components

Both residual methods use 287-feature lane-aware observations and the same final physical action bounds. Their architectures differ: the direct actor has a dedicated history encoder/fusion MLP; SAC uses a two-layer 128-unit actor with squashed Gaussian training outputs and deterministic mean at inference.

Upstream TCP supplies perception, trajectory prediction, route conditioning and PID. The integration changes standalone actor binding, logging, verified control readback, the documented autonomous-speed throttle-cap option and a shared terminal stop. The compressed TCP checkpoint contains the same network tensors.

