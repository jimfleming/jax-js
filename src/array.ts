import { AluExp, DType } from "./alu";
import { Backend, Slot } from "./backend";
import { ShapeTracker } from "./shape";

const JsArray = globalThis.Array;

class PendingExecute {
  submitted = false;
  #controller: AbortController | null = null;
  #promise: Promise<void> | null = null;

  constructor(
    readonly exp: AluExp,
    readonly inputs: Slot[],
    readonly outputs: Slot[],
  ) {}

  async submit(backend: Backend) {
    if (this.submitted) return;
    if (this.#promise) {
      await this.#promise;
      return;
    }
    this.#controller = new AbortController();
    this.#promise = backend.execute(
      this.exp,
      this.inputs,
      this.outputs,
      this.#controller.signal,
    );
    await this.#promise;
    this.submitted = true;
  }

  submitSync(backend: Backend) {
    if (this.submitted) return;
    if (this.#controller) this.#controller.abort();
    backend.executeSync(this.exp, this.inputs, this.outputs);
    this.submitted = true;
  }
}

export class Array {
  shape: number[];
  dtype: DType;

  #source: AluExp | Slot;
  #st: ShapeTracker;
  #backend: Backend;
  #pending: Set<PendingExecute> | null;

  constructor(
    source: AluExp | Slot,
    st: ShapeTracker,
    dtype: DType,
    backend: Backend,
    pending: Set<PendingExecute> | null = null,
  ) {
    this.shape = st.shape;
    this.dtype = dtype;

    this.#source = source;
    this.#st = st;
    this.#backend = backend;
    this.#pending = pending;
  }

  static zeros(shape: number[], dtype: DType) {
    return new Array(
      AluExp.const(dtype, 0),
      ShapeTracker.fromShape(shape),
      dtype,
      backend,
    );
  }

  static ones(shape: number[], dtype: DType) {
    return new Array(
      AluExp.const(dtype, 1),
      ShapeTracker.fromShape(shape),
      dtype,
      backend,
    );
  }

  add(other: Array) {
    return this._binary(AluExp.add, other);
  }
}
