import {
  grad,
  gradOpts,
  jacfwd,
  jacrev,
  jit,
  jvp,
  linearize,
  makeJaxpr,
  nn,
  numpy as np,
  tree,
  valueAndGrad,
  vjp,
  vmap,
} from "@jax-js/jax";
import { expect, suite, test } from "vitest";

suite("jax.makeJaxpr()", () => {
  test("tracks a nullary function", () => {
    const { jaxpr } = makeJaxpr(() => np.multiply(2, 2))();
    expect(jaxpr.toString()).toMatchInlineSnapshot(`
      "{ lambda  .
        ( 4 ) }"
    `);
    expect(jaxpr.consts).toEqual([]);
  });

  test("tracks a unary function", () => {
    const { jaxpr } = makeJaxpr((x: np.Array) => np.multiply(x.add(2), x))(
      np.array([
        [2, 4, 10],
        [1, 1, 1],
      ]),
    );
    expect(jaxpr.toString()).toMatchInlineSnapshot(`
      "{ lambda a:float32[2,3] .
        let b:float32[2,3] = add a 2
            c:float32[2,3] = mul b a
        in ( c ) }"
    `);
    expect(jaxpr.consts).toEqual([]);
  });

  test("composes with jvp", () => {
    const f = (x: np.Array) => np.multiply(x.add(2), x);
    const fdot = (x: np.Array) => jvp(f, [x], [1])[1];

    const { jaxpr } = makeJaxpr(fdot)(np.array(2));
    expect(jaxpr.toString()).toMatchInlineSnapshot(`
      "{ lambda a:float32[] .
        let b:float32[] = add a 2
            c:float32[] = add b a
        in ( c ) }"
    `);
    expect(jaxpr.consts).toEqual([]);
  });

  test("composes with grad", () => {
    const f = (x: np.Array) => {
      const y = x.ref.add(2);
      return x.ref.mul(x).add(y);
    };
    const { jaxpr } = makeJaxpr(grad(f))(3);
    expect(jaxpr.consts).toEqual([]);
    expect(jaxpr.toString()).toMatchInlineSnapshot(`
      "{ lambda a:float32[] .
        let b:float32[] = add 1 a
            c:float32[] = add b a
        in ( c ) }"
    `);
  });

  test("can flatten() nested Jaxprs", () => {
    const f = (x: np.Array) => {
      const y = x.add(2);
      return x.mul(x).add(y);
    };
    const jf = jit(f);

    const { jaxpr } = makeJaxpr((x) => f(jf(x)))(3);
    expect(jaxpr.consts).toEqual([]);
    expect(jaxpr.toString()).toMatchInlineSnapshot(`
      "{ lambda a:float32[] .
        let b:float32[] = jit [ name=f
                                jaxpr={ lambda a:float32[] .
                                  let b:float32[] = add a 2
                                      c:float32[] = mul a a
                                      d:float32[] = add c b
                                  in ( d ) }
                                numConsts=0 ] a
            c:float32[] = add b 2
            d:float32[] = mul b b
            e:float32[] = add d c
        in ( e ) }"
    `);
    expect(jaxpr.jaxpr.flatten().toString()).toMatchInlineSnapshot(`
      "{ lambda a:float32[] .
        let b:float32[] = add a 2
            c:float32[] = mul a a
            d:float32[] = add c b
            e:float32[] = add d 2
            f:float32[] = mul d d
            g:float32[] = add f e
        in ( g ) }"
    `);
  });
});

suite("jax.linearize()", () => {
  test("works for scalars", () => {
    const [y, lin] = linearize(np.sin, 3);
    expect(y).toBeAllclose(np.sin(3));
    expect(lin(1)).toBeAllclose(np.cos(3));
    expect(lin(-42)).toBeAllclose(np.cos(3).mul(-42));
  });

  test("works for simple arrays", () => {
    const [y, lin] = linearize((x: np.Array) => x.ref.mul(x), np.array([2, 3]));
    expect(y).toBeAllclose(np.array([4, 9]));
    expect(lin(np.array([1, 0]))).toBeAllclose(np.array([4, 0]));
    expect(lin(np.array([0, 1]))).toBeAllclose(np.array([0, 6]));
  });

  test("can take and return jstrees", () => {
    const [y, lin] = linearize(
      (x: { a: np.Array; b: np.Array }) => ({
        r1: x.a.ref.mul(x.a).add(x.b.ref),
        r2: x.b,
      }),
      { a: 1, b: 2 },
    );
    expect(y.r1).toBeAllclose(3);
    expect(y.r2).toBeAllclose(2);

    const { r1: r1Dot, r2: r2Dot } = lin({ a: 1, b: 0 });
    expect(r1Dot).toBeAllclose(2);
    expect(r2Dot).toBeAllclose(0);
  });
});

suite("jax.vjp()", () => {
  test("works for scalars", () => {
    const [y, backward] = vjp(np.sin, 3);
    expect(y).toBeAllclose(np.sin(3));
    expect(backward(1)[0]).toBeAllclose(np.cos(3));
  });
});

suite("jax.grad()", () => {
  test("works for a simple scalar function", () => {
    const f = (x: np.Array) => x.ref.mul(x.ref).mul(x); // d/dx (x^3) = 3x^2
    const df = grad(f);
    expect(df(4)).toBeAllclose(48);
    expect(df(5)).toBeAllclose(75);
    expect(df(0)).toBeAllclose(0);
    expect(df(-4)).toBeAllclose(48);
  });

  test("can compute higher derivatives", () => {
    const f = (x: np.Array) => np.sin(np.cos(x));
    const df = grad(f); // d/dx sin(cos(x)) = -sin(x)cos(cos(x))
    const ddf = grad(df); // d^2/dx^2 sin(cos(x)) = -sin^2(x)sin(cos(x)) - cos(x)cos(cos(x))
    expect(df(3)).toBeAllclose(-0.077432003);
    expect(ddf(3)).toBeAllclose(0.559854311);
  });

  test("can compute grad of products", () => {
    const x = np.array([1, 2, 3, 4]);
    const gradProd = grad((x: np.Array) => np.prod(x))(x);
    expect(gradProd.js()).toEqual([24, 12, 8, 6]);
  });

  test("backprops through auto-broadcast", () => {
    const [dx, dy] = grad(([x, y]: [np.Array, np.Array]) => x.mul(y).sum())([
      np.array([[2], [4]]),
      np.array([4, 5, 6]),
    ]);
    expect(dx.js()).toEqual([[15], [15]]);
    expect(dy.js()).toEqual([6, 6, 6]);

    const [dx2, dy2] = grad(([x, y]: [np.Array, np.Array]) => x.add(y).sum())([
      np.array([[2], [4]]),
      np.array([4, 5, 6]),
    ]);
    expect(dx2.js()).toEqual([[3], [3]]);
    expect(dy2.js()).toEqual([2, 2, 2]);
  });

  test("backprop handles dense layer", () => {
    type Params = {
      w: np.Array;
      b: np.Array;
    };

    // x is of shape [batch, dim_in]
    // params.w is of shape [dim_in, dim_out]
    // params.b is of shape [dim_out]

    // const dense = (params: Params, x: np.Array) =>
    //   np
    //     .multiply(x.reshape([...x.shape, 1]), params.w)
    //     .sum(-2)
    //     .add(params.b);

    const dense = (params: Params, x: np.Array) =>
      np.dot(x, params.w).add(params.b);

    const loss = (params: Params, x: np.Array) =>
      nn.logSoftmax(dense(params, x)).slice([], 0).sum().mul(0.5);

    const params: Params = {
      w: np.array([
        [0.1, 0.2, -0.3, 0.0],
        [0.5, -0.1, 0.3, 0.4],
      ]),
      b: np.array([0, 0, 0, 0]),
    };
    const x = np.array([
      [0.1, 0.2],
      [0.2, 0.3],
    ]);

    // These numbers are checked for consistency with JAX.
    expect(loss(tree.ref(params), x.ref)).toBeAllclose(-1.3165712);

    // console.log(makeJaxpr(grad(loss))(params, x).jaxpr.toString());
    const grads = grad(loss)(params, x);
    expect(grads.w).toBeAllclose([
      [0.1095776, -0.03502218, -0.03585088, -0.03870453],
      [0.18276905, -0.05844341, -0.05986644, -0.06445917],
    ]);
    expect(grads.b).toBeAllclose([
      0.7319144, -0.2342123, -0.24015564, -0.25754642,
    ]);
  });
});

suite("jax.jit()", () => {
  test("works for a simple scalar function", () => {
    const f = (x: np.Array) => x.ref.mul(x.ref).mul(x); // d/dx (x^3) = 3x^2
    const f2 = jit(f);
    expect(f(np.array(2))).toBeAllclose(8);
    expect(f2(np.array(2))).toBeAllclose(8);
  });

  test("works with identity function", () => {
    const f = jit((x: np.Array) => x);
    const a = f(np.array(3));
    expect(a.js()).toEqual(3);
  });

  test("works with duplicate output", () => {
    const f = jit((x: np.Array) => [x.ref, x]);
    const [a, b] = f(np.array(3));
    expect(a.js()).toEqual(3);
    expect(b.js()).toEqual(3);
  });

  test("processes gather ops", () => {
    const f = jit((x: np.Array) =>
      x.slice(np.array([1, 3, 2, 0], { dtype: np.int32 })),
    );
    const a = f(np.array([10, 20, 30, 40]));
    expect(a.js()).toEqual([20, 40, 30, 10]);
  });

  test("supports staticArgnums", () => {
    const f = jit((x: np.Array, idx: number) => x.slice(idx), {
      staticArgnums: [1],
    });
    expect(f(np.arange(20), 0).js()).toEqual(0);
    expect(f(np.arange(20), 3).js()).toEqual(3);
    expect(f(np.array([30, 1, 20, 11]), 3).js()).toEqual(11);
  });

  test("jit-of-jit", () => {
    const f = jit((x: np.Array) => x.ref.mul(x));
    const g = jit((x: np.Array) => f(f(x)));
    expect(g(3)).toBeAllclose(81);
    expect(jit(jit(g))(3)).toBeAllclose(81);
  });

  test("jvp-of-jit", () => {
    const f = jit((x: np.Array) => x.ref.mul(x));
    expect(jvp(f, [3], [1])).toBeAllclose([9, 6]);
  });

  test("grad-of-jit", () => {
    const f = jit((x: np.Array) => x.ref.mul(x));

    expect(grad(f)(3)).toBeAllclose(6);
    expect(grad(f)(10)).toBeAllclose(20);
    expect(jvp(grad(f), [10], [1])).toBeAllclose([20, 2]);
    expect(grad(grad(f))(10)).toBeAllclose(2);
    expect(grad(jit(grad(f)))(10)).toBeAllclose(2);
  });

  test("vmap-of-jit", () => {
    const s = jit((x: np.Array) => x.sum());
    const ar = np.array([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    expect(s(ar.ref).js()).toEqual(21);
    expect(vmap(s)(ar.ref).js()).toEqual([6, 15]);
    expect(vmap(s, 1)(ar).js()).toEqual([5, 7, 9]);
  });
});

suite("hasAux parameter", () => {
  test("jvp with hasAux", () => {
    // Use .ref before each reuse of x (move semantics)
    const f = (x: np.Array) =>
      [x.ref.mul(x.ref), x.sum()] as [np.Array, np.Array];
    const [[y, aux], dy] = jvp(
      f,
      [np.array([2, 3])],
      [np.array([1, 0])],
      gradOpts({ hasAux: true }),
    );
    expect(y).toBeAllclose(np.array([4, 9]));
    expect(aux).toBeAllclose(5);
    expect(dy).toBeAllclose(np.array([4, 0]));
  });

  test("vjp with hasAux", () => {
    const f = (x: np.Array) =>
      [x.ref.mul(x.ref), x.sum()] as [np.Array, np.Array];
    const [y, backward, aux] = vjp(
      f,
      gradOpts({ hasAux: true }),
      np.array([2, 3]),
    );
    expect(y).toBeAllclose(np.array([4, 9]));
    expect(aux).toBeAllclose(5);
    expect(backward(np.array([1, 1]))[0]).toBeAllclose(np.array([4, 6]));
  });

  test("linearize with hasAux", () => {
    const f = (x: np.Array) =>
      [x.ref.mul(x.ref), x.sum()] as [np.Array, np.Array];
    const [[y, aux], lin] = linearize(
      f,
      gradOpts({ hasAux: true }),
      np.array([2, 3]),
    );
    expect(y).toBeAllclose(np.array([4, 9]));
    expect(aux).toBeAllclose(5);
    expect(lin(np.array([1, 0]))).toBeAllclose(np.array([4, 0]));
  });

  test("grad with hasAux", () => {
    const f = (x: np.Array) =>
      [x.ref.mul(x.ref).sum(), x.mul(2)] as [np.Array, np.Array];
    const [dx, aux] = grad(f, gradOpts({ hasAux: true }))(np.array([2, 3]));
    expect(dx).toBeAllclose(np.array([4, 6]));
    expect(aux).toBeAllclose(np.array([4, 6]));
  });

  test("valueAndGrad with hasAux", () => {
    const f = (x: np.Array) =>
      [x.ref.mul(x.ref).sum(), x.mul(2)] as [np.Array, np.Array];
    const [[y, aux], dx] = valueAndGrad(f, gradOpts({ hasAux: true }))(
      np.array([2, 3]),
    );
    expect(y).toBeAllclose(13);
    expect(aux).toBeAllclose(np.array([4, 6]));
    expect(dx).toBeAllclose(np.array([4, 6]));
  });

  test("jacrev with hasAux", () => {
    const f = (x: np.Array) =>
      [x.ref.mul(x.ref), x.sum()] as [np.Array, np.Array];
    const [jac, aux] = jacrev(f, gradOpts({ hasAux: true }))(np.array([2, 3]));
    expect(jac).toBeAllclose(
      np.array([
        [4, 0],
        [0, 6],
      ]),
    );
    expect(aux).toBeAllclose(5);
  });

  test("jacfwd with hasAux", () => {
    const f = (x: np.Array) =>
      [x.ref.mul(x.ref), x.sum()] as [np.Array, np.Array];
    const [jac, aux] = jacfwd(f, gradOpts({ hasAux: true }))(np.array([2, 3]));
    expect(jac).toBeAllclose(
      np.array([
        [4, 0],
        [0, 6],
      ]),
    );
    expect(aux).toBeAllclose(5);
  });

  test("grad with hasAux and nested tree aux", () => {
    // Test hasAux with nested tree structure as auxiliary output
    const f = (x: np.Array) =>
      [
        x.ref.mul(x.ref).sum(),
        { mean: x.ref.mean(), parts: [np.min(x.ref), np.max(x)] },
      ] as [np.Array, { mean: np.Array; parts: np.Array[] }];
    const [dx, aux] = grad(f, gradOpts({ hasAux: true }))(np.array([2, 3, 4]));
    expect(dx).toBeAllclose(np.array([4, 6, 8]));
    expect(aux.mean).toBeAllclose(3);
    expect(aux.parts[0]).toBeAllclose(2);
    expect(aux.parts[1]).toBeAllclose(4);
  });

  test("jit composed with grad and hasAux", () => {
    // Test composition: jit(grad(f, gradOpts({ hasAux: true })))
    const f = (x: np.Array) =>
      [x.ref.mul(x.ref).sum(), x.mul(2)] as [np.Array, np.Array];
    const jitGrad = jit(grad(f, gradOpts({ hasAux: true })));
    const [dx, aux] = jitGrad(np.array([2, 3]));
    expect(dx).toBeAllclose(np.array([4, 6]));
    expect(aux).toBeAllclose(np.array([4, 6]));
  });

  test("jit composed with valueAndGrad and hasAux", () => {
    // Test composition: jit(valueAndGrad(f, gradOpts({ hasAux: true })))
    const f = (x: np.Array) =>
      [x.ref.mul(x.ref).sum(), x.mul(2)] as [np.Array, np.Array];
    const jitValueAndGrad = jit(valueAndGrad(f, gradOpts({ hasAux: true })));
    const [[y, aux], dx] = jitValueAndGrad(np.array([2, 3]));
    expect(y).toBeAllclose(13);
    expect(aux).toBeAllclose(np.array([4, 6]));
    expect(dx).toBeAllclose(np.array([4, 6]));
  });
});
