import { grad, numpy as np } from "@jax-js/jax";
import { adamw, applyUpdates, sgd, squaredError } from "@jax-js/optax";
import { expect, test } from "vitest";

test("adamw optimizer", () => {
  let params = np.array([1.0, 2.0, 3.0]);

  const solver = adamw(0.001);
  let optState = solver.init(params.ref);
  let updates: np.Array;

  const f = (x: np.Array) => squaredError(x, np.ones([3])).sum();
  const paramsGrad = grad(f)(params.ref);
  [updates, optState] = solver.update(paramsGrad, optState, params.ref);
  params = applyUpdates(params, updates);

  expect(params.shape).toEqual([3]);
  expect(params.dtype).toEqual(np.float32);
});

test("adamw with custom weight decay", () => {
  let params = np.array([1.0, 2.0, 3.0]);

  const solver = adamw(0.001, { weightDecay: 0.01 });
  let optState = solver.init(params.ref);
  let updates: np.Array;

  const f = (x: np.Array) => squaredError(x, np.ones([3])).sum();
  const paramsGrad = grad(f)(params.ref);
  [updates, optState] = solver.update(paramsGrad, optState, params.ref);
  params = applyUpdates(params, updates);

  expect(params.shape).toEqual([3]);
  expect(params.dtype).toEqual(np.float32);
});

test("adamw with nesterov", () => {
  let params = np.array([1.0, 2.0, 3.0]);

  const solver = adamw(0.001, { nesterov: true, weightDecay: 0.005 });
  let optState = solver.init(params.ref);
  let updates: np.Array;

  const f = (x: np.Array) => squaredError(x, np.ones([3])).sum();
  const paramsGrad = grad(f)(params.ref);
  [updates, optState] = solver.update(paramsGrad, optState, params.ref);
  params = applyUpdates(params, updates);

  expect(params.shape).toEqual([3]);
  expect(params.dtype).toEqual(np.float32);
});
