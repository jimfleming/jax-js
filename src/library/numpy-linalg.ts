import * as lax from "./lax";
import { triangularSolve } from "./lax-linalg";
import { Array, ArrayLike, matmul, matrixTranspose } from "./numpy";
import { fudgeArray } from "../frontend/array";

/**
 * Compute the Cholesky decomposition of a (batched) positive-definite matrix.
 *
 * This is like `jax.lax.linalg.cholesky()`, except with an option to symmetrize
 * the input matrix, which is on by default.
 */
export function cholesky(
  a: ArrayLike,
  {
    upper = false,
    symmetrizeInput = true,
  }: {
    upper?: boolean;
    symmetrizeInput?: boolean;
  } = {},
): Array {
  a = fudgeArray(a);
  if (a.ndim < 2 || a.shape[a.ndim - 1] !== a.shape[a.ndim - 2]) {
    throw new Error(
      `cholesky: input must be at least 2D square matrix, got ${a.aval}`,
    );
  }
  if (symmetrizeInput) {
    a = a.ref.add(matrixTranspose(a)).mul(0.5);
  }
  return lax.linalg.cholesky(a, { upper });
}

export { diagonal } from "./numpy";

/**
 * Return the least-squares solution to a linear equation.
 *
 * For overdetermined systems, this finds the `x` that minimizes `norm(ax - b)`.
 * For underdetermined systems, this finds the minimum-norm solution for `x`.
 *
 * This currently uses Cholesky decomposition to solve the normal equations,
 * under the hood. The method is not as robust as QR or SVD.
 *
 * @param a coefficient matrix of shape `(M, N)`
 * @param b right-hand side of shape `(M,)` or `(M, K)`
 * @return least-squares solution of shape `(N,)` or `(N, K)`
 */
export function lstsq(a: ArrayLike, b: ArrayLike): Array {
  a = fudgeArray(a);
  b = fudgeArray(b);
  if (a.ndim !== 2)
    throw new Error(`lstsq: 'a' must be a 2D array, got ${a.aval}`);
  const [m, n] = a.shape;
  if (b.shape[0] !== m)
    throw new Error(
      `lstsq: leading dimension of 'b' must match number of rows of 'a', got ${b.aval}`,
    );
  const at = matrixTranspose(a.ref);
  if (m <= n) {
    // Underdetermined or square system: A.T @ (A @ A.T)^-1 @ B
    const aat = matmul(a, at.ref); // A @ A.T, shape (M, M)
    const l = cholesky(aat, { symmetrizeInput: false }); // L @ L.T = A @ A.T
    const lb = triangularSolve(l.ref, b, { leftSide: true, lower: true }); // L^-1 @ B
    const llb = triangularSolve(l, lb, { leftSide: true, transposeA: true }); // (A @ A.T)^-1 @ B
    return matmul(at, llb.ref); // A.T @ (A @ A.T)^-1 @ B
  } else {
    // Overdetermined system: (A.T @ A)^-1 @ A.T @ B
    const ata = matmul(at.ref, a); // A.T @ A, shape (N, N)
    const l = cholesky(ata, { symmetrizeInput: false }); // L @ L.T = A.T @ A
    const atb = matmul(at, b); // A.T @ B
    const lb = triangularSolve(l.ref, atb, { leftSide: true, lower: true }); // L^-1 @ A.T @ B
    const llb = triangularSolve(l, lb, { leftSide: true, transposeA: true }); // (A.T @ A)^-1 @ A.T @ B
    return llb;
  }
}

export { matmul } from "./numpy";
export { matrixTranspose } from "./numpy";
export { outer } from "./numpy";
export { tensordot } from "./numpy";
export { trace } from "./numpy";
export { vecdot } from "./numpy";
