import { GradientTransformation, identity, ScalarOrSchedule } from "./base";
import { chain } from "./combine";
import {
  addDecayedWeights,
  AddDecayedWeightsOptions,
  scaleByAdam,
  ScaleByAdamOptions,
  scaleByLearningRate,
  trace,
  TraceOptions,
} from "./transform";
import { numpy as np } from "@jax-js/jax";

export type SgdOptions = {
  momentum?: number | null;
  nesterov?: boolean;
  accumulatorDtype?: np.DType;
};

/** 
 * A canonical Stochastic Gradient Descent optimizer.
 * 
 * This implements stochastic gradient descent. It also includes support for
 * momentum, and Nesterov acceleration, as these are standard practice when
 * using stochastic gradient descent to train deep neural networks.
 * 
 * Args:
 *   learningRate: A global scaling factor, either fixed or evolving along
 *     iterations with a scheduler.
 *   momentum: Decay rate used by the momentum term, when it is set to null,
 *     then momentum is not used at all.
 *   nesterov: Whether Nesterov momentum is used.
 *   accumulatorDtype: Optional dtype to be used for the accumulator.
 */
export function sgd(
  learningRate: ScalarOrSchedule,
  opts: SgdOptions = {}
): GradientTransformation {
  const { momentum = null, nesterov = false, accumulatorDtype } = opts;
  
  let opt: GradientTransformation;
  if (momentum !== null) {
    opt = trace({
      decay: momentum,
      nesterov,
      accumulatorDtype,
    });
  } else {
    opt = identity();
  }
  
  return chain(opt, scaleByLearningRate(learningRate));
}

/** The Adam optimizer. */
export function adam(
  learningRate: ScalarOrSchedule,
  opts?: ScaleByAdamOptions,
): GradientTransformation {
  return chain(scaleByAdam(opts), scaleByLearningRate(learningRate));
}

export type AdamWOptions = ScaleByAdamOptions & AddDecayedWeightsOptions;

/** 
 * Adam with weight decay regularization.
 * 
 * AdamW uses weight decay to regularize learning towards small weights, as
 * this leads to better generalization. In SGD you can also use L2 regularization
 * to implement this as an additive loss term, however L2 regularization
 * does not behave as intended for adaptive gradient algorithms such as Adam.
 * 
 * Args:
 *   learningRate: A global scaling factor, either fixed or evolving along
 *     iterations with a scheduler.
 *   b1: Exponential decay rate to track the first moment of past gradients.
 *   b2: Exponential decay rate to track the second moment of past gradients.
 *   eps: A small constant applied to denominator outside of the square root
 *     (as in the Adam paper) to avoid dividing by zero when rescaling.
 *   epsRoot: A small constant applied to denominator inside the square root (as
 *     in RMSProp), to avoid dividing by zero when rescaling.
 *   nesterov: Whether to use Nesterov momentum.
 *   weightDecay: Strength of the weight decay regularization. Note that this
 *     weight decay is multiplied with the learning rate.
 *   mask: A tree with same structure as the params PyTree. The leaves
 *     should be arrays with values 0 or 1, where 1 means apply weight decay
 *     and 0 means skip weight decay for that parameter. Note that the Adam 
 *     gradient transformations are applied to all parameters.
 */
export function adamw(
  learningRate: ScalarOrSchedule,
  opts: AdamWOptions = {}
): GradientTransformation {
  const {
    b1,
    b2,
    eps,
    epsRoot,
    nesterov,
    weightDecay = 1e-4,
    mask,
    ...adamOpts
  } = opts;

  return chain(
    scaleByAdam({
      b1,
      b2,
      eps,
      epsRoot,
      nesterov,
      ...adamOpts,
    }),
    addDecayedWeights({ weightDecay, mask }),
    scaleByLearningRate(learningRate)
  );
}
