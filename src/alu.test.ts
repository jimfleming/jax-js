import { expect, test } from "vitest";
import { AluExp } from "./alu";

test("AluExp can be evaluated", () => {
  const e = AluExp.const(3);
  expect(e.evaluate({})).toEqual(3);

  const e2 = AluExp.add(AluExp.const(3), AluExp.const(4));
  expect(e2.evaluate({})).toEqual(7);

  const e3 = AluExp.add(AluExp.const(3), e2);
  expect(e3.evaluate({})).toEqual(10);

  const e4 = AluExp.mul(AluExp.special("idx", 10), AluExp.const(50));
  expect(e4.evaluate({ idx: 10 })).toEqual(500);
});

test("AluExp works with ternaries", () => {
  const x = AluExp.special("x", 100);

  const e = AluExp.where(
    AluExp.cmplt(x, AluExp.const(70)),
    AluExp.const(0),
    AluExp.const(1),
  );
  expect(e.evaluate({ x: 50 })).toEqual(0);
  expect(e.evaluate({ x: 69 })).toEqual(0);
  expect(e.evaluate({ x: 70 })).toEqual(1);
  expect(e.evaluate({ x: 80 })).toEqual(1);
});
