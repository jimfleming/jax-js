/** Mathemtical expression on scalar values. */
export class AluExp {
  // TODO: Currently untyped, make this dtype-aware.
  constructor(
    readonly op: AluOp,
    readonly src: AluExp[],
    readonly arg: any = undefined,
  ) {}

  static add(a: AluExp, b: AluExp): AluExp {
    return new AluExp(AluOp.Add, [a, b]);
  }
  static sub(a: AluExp, b: AluExp): AluExp {
    return new AluExp(AluOp.Sub, [a, b]);
  }
  static mul(a: AluExp, b: AluExp): AluExp {
    return new AluExp(AluOp.Mul, [a, b]);
  }
  static idiv(a: AluExp, b: AluExp): AluExp {
    return new AluExp(AluOp.Idiv, [a, b]);
  }
  static mod(a: AluExp, b: AluExp): AluExp {
    return new AluExp(AluOp.Mod, [a, b]);
  }
  static cmplt(a: AluExp, b: AluExp): AluExp {
    return new AluExp(AluOp.Cmplt, [a, b]);
  }
  static cmpne(a: AluExp, b: AluExp): AluExp {
    return new AluExp(AluOp.Cmpne, [a, b]);
  }
  static where(cond: AluExp, a: AluExp, b: AluExp): AluExp {
    return new AluExp(AluOp.Where, [cond, a, b]);
  }
  static const(value: any): AluExp {
    return new AluExp(AluOp.Const, [], value);
  }
  static special(name: string, n: number): AluExp {
    return new AluExp(AluOp.Special, [], [name, n]);
  }

  /** Evaluate the expression, returning the result (or a list of results). */
  evaluate(context: Record<string, any>): any {
    switch (this.op) {
      case AluOp.Add:
        return this.src[0].evaluate(context) + this.src[1].evaluate(context);
      case AluOp.Sub:
        return this.src[0].evaluate(context) - this.src[1].evaluate(context);
      case AluOp.Mul:
        return this.src[0].evaluate(context) * this.src[1].evaluate(context);
      case AluOp.Idiv:
        return Math.floor(
          this.src[0].evaluate(context) / this.src[1].evaluate(context),
        );
      case AluOp.Mod:
        return this.src[0].evaluate(context) % this.src[1].evaluate(context);
      case AluOp.Cmplt:
        return this.src[0].evaluate(context) < this.src[1].evaluate(context);
      case AluOp.Cmpne:
        return this.src[0].evaluate(context) !== this.src[1].evaluate(context);
      case AluOp.Where:
        return this.src[0].evaluate(context)
          ? this.src[1].evaluate(context)
          : this.src[2].evaluate(context);
      case AluOp.Const:
        return this.arg;
      case AluOp.Special:
        return context[this.arg[0]];
    }
  }
}

/** Symbolic form for each mathematical operation. */
export enum AluOp {
  Add,
  Sub,
  Mul,
  Idiv,
  Mod,
  Cmplt,
  Cmpne,
  Where,
  Const, // arg = value
  Special, // arg = [variable_name, n]
}
